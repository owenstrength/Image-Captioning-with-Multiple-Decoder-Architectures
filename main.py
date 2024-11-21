
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
import matplotlib.pyplot as plt

# Define the CNN Encoder (unchanged)
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.linear = nn.Linear(2048, embed_size)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.view(features.size(0), -1)
        return self.linear(features)

# Modified Decoder with fixes
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        
        self.dropout = nn.Dropout(0.5)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.5 if num_layers > 1 else 0)
        
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, features, captions):
        batch_size = features.size(0)
        
        # Embed captions
        embeddings = self.embedding(captions[:, :-1])  # Remove last token for input
        embeddings = self.dropout(embeddings)
        
        # Prepare features
        features = features.unsqueeze(1)
        
        # Concatenate features with embeddings
        inputs = torch.cat((features, embeddings), dim=1)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(inputs)
        
        # Process through fully connected layers
        outputs = self.dropout(lstm_out)
        outputs = torch.relu(self.fc1(outputs))
        outputs = self.fc2(outputs)
        
        return outputs  # Shape will be [batch_size, sequence_length, vocab_size]

    def sample(self, features, tokenizer, max_len=20):
        self.eval()
        
        batch_size = features.size(0)
        sampled_ids = []
        states = None
        
        # Start with features
        inputs = features.unsqueeze(1)
        
        # Start token
        input_word = torch.tensor([[tokenizer.cls_token_id]] * batch_size).to(features.device)
        
        for i in range(max_len):
            # For first token, use features. For subsequent tokens, use word embedding
            if i == 0:
                curr_inputs = inputs
            else:
                curr_inputs = self.embedding(input_word)
            
            # Forward pass
            lstm_out, states = self.lstm(curr_inputs, states)
            outputs = self.fc1(lstm_out.squeeze(1))
            outputs = self.fc2(outputs)
            
            # Sample next word
            predicted = outputs.argmax(1)
            sampled_ids.append(predicted.item())
            
            # Break if end token is predicted
            if predicted.item() == tokenizer.sep_token_id:
                break
                
            input_word = predicted.unsqueeze(1)
        
        return torch.LongTensor(sampled_ids)

# Modified Dataset Class for COCO with subset sampling
class CocoDataset(Dataset):
    def __init__(self, coco_json, tokenizer, transform=None, img_dir='', subset_fraction=0.001):
        self.coco = COCO(coco_json)
        all_image_ids = self.coco.getImgIds()
        
        # Calculate number of images for the subset
        subset_size = max(1, int(len(all_image_ids) * subset_fraction))
        
        # Randomly sample image IDs
        self.image_ids = random.sample(all_image_ids, subset_size)
        
        print(f"Using {len(self.image_ids)} images out of {len(all_image_ids)} total images")
        
        self.transform = transform
        self.tokenizer = tokenizer
        self.img_dir = img_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        
        image_path = f"{self.img_dir}/{img_info['file_name']}"
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        img_captions = self.coco.imgToAnns[img_id]
        caption = img_captions[0]['caption']
        caption_tokens = self.tokenizer(caption, padding='max_length', max_length=128, 
                                      return_tensors='pt', truncation=True)
        return image, caption_tokens['input_ids'].squeeze(0)

# Modified data loading function
def load_data(coco_json, img_dir, subset_fraction=0.001):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = CocoDataset(coco_json, tokenizer, transform, img_dir, subset_fraction)
    
    def collate_fn(batch):
        images, captions = zip(*batch)
        images = torch.stack(images, 0)
        captions = torch.stack(captions, 0)
        return images, captions
    
    return DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn), tokenizer


def generate_sample_caption(encoder, decoder, image_tensor, tokenizer, device):
    """Generate a caption for a sample image during training"""
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        # Encode image
        features = encoder(image_tensor.unsqueeze(0))
        
        # Initialize caption generation
        caption = []
        input_word = torch.tensor([[tokenizer.cls_token_id]]).to(device)
        
        # Generate caption word by word
        for _ in range(30):  # Max length of 30 words
            embeddings = decoder.embedding(input_word)
            if len(caption) == 0:
                lstm_input = torch.cat((features.unsqueeze(1), embeddings), 1)
            else:
                lstm_input = embeddings
            
            lstm_out, _ = decoder.lstm(lstm_input)
            outputs = decoder.fc(lstm_out)
            
            # Get the most likely next word
            predicted = outputs.argmax(2)
            predicted_token_id = predicted[0, -1].item()
            
            if predicted_token_id == tokenizer.sep_token_id:
                break
                
            caption.append(predicted_token_id)
            input_word = predicted
    
    return tokenizer.decode(caption, skip_special_tokens=True)

def visualize_sample(image_tensor, actual_caption, predicted_caption, epoch):
    """Display the image with actual and predicted captions"""
    image = image_tensor.cpu().clone()
    # Denormalize the image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = image.clamp(0, 1)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(image.permute(1, 2, 0))
    plt.axis('off')
    plt.title(f'Sample Prediction - Epoch {epoch}', pad=20)
    plt.figtext(0.1, 0.05, f'Actual: {actual_caption}', wrap=True, fontsize=10)
    plt.figtext(0.1, 0.02, f'Predicted: {predicted_caption}', wrap=True, fontsize=10)
    plt.show()

def train_model(encoder, decoder, data_loader, tokenizer, num_epochs=5, device='cuda'):
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    params = list(decoder.parameters()) + list(encoder.linear.parameters())
    optimizer = optim.AdamW(params, lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        total_loss = 0
        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for i, (images, captions) in enumerate(progress_bar):
            images = images.to(device)
            captions = captions.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            features = encoder(images)
            
            outputs = decoder(features, captions)
            
            # Calculate loss
            targets = captions[:, 1:]  # Remove first token (CLS) from target
            outputs = outputs[:, :targets.size(1), :]  # Match sequence length
            loss = criterion(outputs.reshape(-1, decoder.vocab_size), targets.reshape(-1))
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)  # Fixed clipping function
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
            # Print example prediction every 100 batches
            if i % 100 == 0:
                with torch.no_grad():
                    sampled_ids = decoder.sample(features[0:1], tokenizer)
                    predicted_caption = tokenizer.decode(sampled_ids, skip_special_tokens=True)
                    actual_caption = tokenizer.decode(captions[0], skip_special_tokens=True)
                    print(f"\nSample prediction:")
                    print(f"Actual: {actual_caption}")
                    print(f"Predicted: {predicted_caption}")
        
        scheduler.step()
        
        avg_loss = total_loss / len(data_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint for best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss
            }
            torch.save(checkpoint, 'checkpoints/best_checkpoint.pth')
            print("Saved best checkpoint!")



if __name__ == '__main__':
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    coco_json = 'coco_dataset/annotations/captions_train2017.json'
    img_dir = 'coco_dataset/train2017'
    data_loader, tokenizer = load_data(coco_json, img_dir, subset_fraction=0.001)
    
    # Initialize models
    embed_size = 256
    hidden_size = 512
    vocab_size = tokenizer.vocab_size
    
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers=2).to(device)
    
    # Train the model
    train_model(encoder, decoder, data_loader, tokenizer, num_epochs=1000, device=device)
