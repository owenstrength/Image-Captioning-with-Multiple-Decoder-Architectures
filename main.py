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
from models.EncoderCNN import EncoderCNN
from models.DecoderRNN import DecoderRNN
from coco_dataset import CocoDataset


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