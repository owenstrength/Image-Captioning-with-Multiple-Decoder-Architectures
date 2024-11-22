import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from pycocotools.coco import COCO
from PIL import Image
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.optim as optim
import random
import math
import torch.utils.data as data

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Disable learning for all parameters
        for param in resnet.parameters():
            param.requires_grad_(False)
            
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_dim = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))

    def forward(self, features, captions):
        # Remove end token and embed captions
        cap_embedding = self.embed(captions[:, :-1])
        
        # Add image features as first token
        embeddings = torch.cat((features.unsqueeze(1), cap_embedding), dim=1)
        
        # LSTM forward pass
        lstm_out, self.hidden = self.lstm(embeddings)
        outputs = self.linear(lstm_out)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        res = []
        
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            outputs = self.linear(lstm_out.squeeze(1))
            _, predicted_idx = outputs.max(1)
            res.append(predicted_idx.item())
            
            if predicted_idx == 1:  # End token
                break
                
            inputs = self.embed(predicted_idx)
            inputs = inputs.unsqueeze(1)
            
        return res

class CocoDataset(Dataset):
    def __init__(self, coco_json, tokenizer, transform=None, img_dir='', subset_fraction=0.001):
        self.coco = COCO(coco_json)
        all_image_ids = self.coco.getImgIds()
        
        # Calculate number of images for subset
        subset_size = max(1, int(len(all_image_ids) * subset_fraction))
        self.image_ids = random.sample(all_image_ids, subset_size)
        
        print(f"Using {len(self.image_ids)} images out of {len(all_image_ids)} total images")
        
        self.transform = transform
        self.tokenizer = tokenizer
        self.img_dir = img_dir
        
        # Create caption lengths list
        self.caption_lengths = {}
        for img_id in self.image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            self.caption_lengths[img_id] = len(anns[0]['caption'].split())

    def get_train_indices(self):
        sel_length = random.choice(list(self.caption_lengths.values()))
        return [i for i, v in enumerate(self.image_ids) 
                if self.caption_lengths[v] == sel_length]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        
        image_path = f"{self.img_dir}/{img_info['file_name']}"
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        caption = anns[0]['caption']
        
        caption_tokens = self.tokenizer(caption, padding='max_length', max_length=128, 
                                      return_tensors='pt', truncation=True)
        return image, caption_tokens['input_ids'].squeeze(0)

def train_model(encoder, decoder, data_loader, tokenizer, num_epochs=5, device='cuda', print_every=20):
    criterion = nn.CrossEntropyLoss().cuda()
    params = list(decoder.parameters()) + list(encoder.embed.parameters())
    optimizer = optim.Adam(params, lr=0.001)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    
    total_step = math.ceil(len(data_loader.dataset) / data_loader.batch_sampler.batch_size)
    
    for epoch in range(1, num_epochs + 1):
        encoder.train()
        decoder.train()
        total_loss = 0
        progress_bar = tqdm(range(1, total_step + 1), desc=f'Epoch {epoch}/{num_epochs}')
        
        for i_step in progress_bar:
            # Get batch with sampled indices
            indices = data_loader.dataset.get_train_indices()
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            data_loader.batch_sampler.sampler = new_sampler
            
            # Get the batch
            images, captions = next(iter(data_loader))
            images = images.to(device)
            captions = captions.to(device)
            
            # Zero gradients
            decoder.zero_grad()
            encoder.zero_grad()
            
            # Forward pass
            features = encoder(images)
            outputs = decoder(features, captions)
            
            # Calculate loss
            loss = criterion(outputs.view(-1, len(tokenizer)), captions.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
            if i_step % print_every == 0:
                with torch.no_grad():
                    features_sample = features[0:1].unsqueeze(1)
                    sampled_ids = decoder.sample(features_sample)
                    predicted_caption = tokenizer.decode(sampled_ids, skip_special_tokens=True)
                    actual_caption = tokenizer.decode(captions[0], skip_special_tokens=True)
                    print(f"\nSample prediction:")
                    print(f"Actual: {actual_caption}")
                    print(f"Predicted: {predicted_caption}")

        avg_loss = total_loss / total_step
        print(f'Epoch [{epoch}/{num_epochs}], Average Loss: {avg_loss:.4f}')

        #scheduler.step()
        
        # Save checkpoint
        if epoch % 1 == 0:
            torch.save(decoder.state_dict(), f'checkpoints/decoder-{epoch}.pkl')
            torch.save(encoder.state_dict(), f'checkpoints/encoder-{epoch}.pkl')

if __name__ == '__main__':
    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                           (0.229, 0.224, 0.225))
    ])
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Dataset paths
    coco_json = 'coco_dataset/annotations/captions_train2017.json'
    img_dir = 'coco_dataset/train2017'
    
    # Create dataset
    dataset = CocoDataset(coco_json, tokenizer, transform, img_dir, subset_fraction=0.1)
    
    # Create data loader with custom batch sampler
    batch_size = 128
    batch_sampler = data.BatchSampler(
        data.SequentialSampler(dataset),
        batch_size=batch_size,
        drop_last=False
    )
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler)
    
    # Initialize models
    embed_size = 256
    hidden_size = 512
    vocab_size = tokenizer.vocab_size
    
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)
    
    # Train
    train_model(encoder, decoder, data_loader, tokenizer, num_epochs=50, device=device)