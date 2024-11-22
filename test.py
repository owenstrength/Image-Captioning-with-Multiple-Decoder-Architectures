import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
from transformers import BertTokenizer
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

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
        cap_embedding = self.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), cap_embedding), dim=1)
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

def generate_caption(encoder, decoder, image_tensor, tokenizer, device, max_length=20):
    """
    Generate a caption for a given image using the trained encoder and decoder
    """
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        # Encode image
        image_tensor = image_tensor.unsqueeze(0).to(device)
        features = encoder(image_tensor)
        
        # Use decoder's sample method, which was used during training
        sampled_ids = decoder.sample(features.unsqueeze(1))
        
        # Convert indices to words
        caption = tokenizer.decode(sampled_ids, skip_special_tokens=True)
        
    return caption


def visualize_prediction(image, actual_caption, predicted_caption):
    # Create the figure and display
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title('Generated vs Actual Caption', pad=20, fontsize=16, fontweight='bold', loc='center')
    
    # Add captions
    plt.figtext(0.5, 0.05, f'Actual: {actual_caption}', wrap=True, fontsize=12, color='green', fontweight='bold', ha='center')
    plt.figtext(0.5, 0.02, f'Predicted: {predicted_caption}', wrap=True, fontsize=12, color='blue', fontweight='bold', ha='center')
    
    filename = f"./results/prediction_{random.randint(0, 10000)}.png"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # Close the figure to free memory
    
    print(f"Saved prediction to: {filename}")

def test_model(encoder_weights, decoder_weights, test_coco_json, test_img_dir, num_test_samples=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models
    embed_size = 256
    hidden_size = 512
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size
    
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)
        
    encoder.load_state_dict(encoder_weights)
    decoder.load_state_dict(decoder_weights)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_coco = COCO(test_coco_json)
    test_image_ids = test_coco.getImgIds()
    sample_ids = random.sample(test_image_ids, num_test_samples)
    
    for img_id in sample_ids:
        img_info = test_coco.imgs[img_id]
        image_path = f"{test_img_dir}/{img_info['file_name']}"
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        
        ann_ids = test_coco.getAnnIds(imgIds=img_id)
        anns = test_coco.loadAnns(ann_ids)
        actual_caption = anns[0]['caption']
        
        predicted_caption = generate_caption(encoder, decoder, image_tensor, tokenizer, device)
        visualize_prediction(image, actual_caption, predicted_caption)

def calculate_metrics(encoder_weights, decoder_weights, test_coco_json, test_img_dir, num_test_samples=100):
    try:
        nltk.download('punkt')
    except:
        pass
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    embed_size = 256
    hidden_size = 512
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size
    
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)
        
    encoder.load_state_dict(encoder_weights)
    decoder.load_state_dict(decoder_weights)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Changed to match training transform
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_coco = COCO(test_coco_json)
    test_image_ids = test_coco.getImgIds()
    sample_ids = random.sample(test_image_ids, num_test_samples)
    
    bleu_scores = []
    bleu_1_scores = []
    bleu_2_scores = []
    bleu_3_scores = []
    bleu_4_scores = []
    
    for img_id in tqdm(sample_ids, desc="Calculating metrics"):
        try:
            img_info = test_coco.imgs[img_id]
            image_path = f"{test_img_dir}/{img_info['file_name']}"
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image)
            
            # Get all captions for this image as references
            ann_ids = test_coco.getAnnIds(imgIds=img_id)
            anns = test_coco.loadAnns(ann_ids)
            reference_captions = [ann['caption'].lower() for ann in anns]
            references = [word_tokenize(cap) for cap in reference_captions]
            
            # Generate caption
            predicted_caption = generate_caption(encoder, decoder, image_tensor, tokenizer, device)
            candidate = word_tokenize(predicted_caption.lower())
            
            # Calculate BLEU scores with different n-grams
            bleu_1 = sentence_bleu(references, candidate, weights=(1, 0, 0, 0))
            bleu_2 = sentence_bleu(references, candidate, weights=(0.5, 0.5, 0, 0))
            bleu_3 = sentence_bleu(references, candidate, weights=(0.33, 0.33, 0.33, 0))
            bleu_4 = sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25))
            
            bleu_1_scores.append(bleu_1)
            bleu_2_scores.append(bleu_2)
            bleu_3_scores.append(bleu_3)
            bleu_4_scores.append(bleu_4)
            
            # Use BLEU-4 as the main score
            bleu_scores.append(bleu_4)
            
            # Print some examples
            if len(bleu_scores) % 10 == 0:
                print("\nExample caption:")
                print(f"Predicted: {predicted_caption}")
                print(f"Reference: {reference_captions[0]}")
                print(f"BLEU-1: {bleu_1:.4f}")
                visualize_prediction(image, reference_captions[0], predicted_caption)
                
        except Exception as e:
            print(f"Error processing image {img_id}: {str(e)}")
            continue

    avg_bleu_1 = sum(bleu_1_scores) / len(bleu_1_scores)
    avg_bleu_2 = sum(bleu_2_scores) / len(bleu_2_scores)
    avg_bleu_3 = sum(bleu_3_scores) / len(bleu_3_scores)
    avg_bleu_4 = sum(bleu_4_scores) / len(bleu_4_scores)
    
    print(f"\nMetrics on {len(bleu_scores)} test samples:")
    print(f"Average BLEU-1 score: {avg_bleu_1:.4f}")
    print(f"Average BLEU-2 score: {avg_bleu_2:.4f}")
    print(f"Average BLEU-3 score: {avg_bleu_3:.4f}")
    print(f"Average BLEU-4 score: {avg_bleu_4:.4f}")

if __name__ == '__main__':
    # Update paths to your pickle files
    encoder_path = 'checkpoints/encoder-50.pkl'  # Path to encoder weights
    decoder_path = 'checkpoints/decoder-50.pkl'  # Path to decoder weights
    test_coco_json = 'coco_dataset/annotations/captions_val2017.json'
    test_img_dir = 'coco_dataset/val2017'

    encoder_weights = torch.load(encoder_path)
    decoder_weights = torch.load(decoder_path)
    
    print("Generating sample predictions...")
    test_model(encoder_weights, decoder_weights, test_coco_json, test_img_dir, num_test_samples=5)
    
    print("\nCalculating metrics...")
    calculate_metrics(encoder_weights, decoder_weights, test_coco_json, test_img_dir, num_test_samples=100)