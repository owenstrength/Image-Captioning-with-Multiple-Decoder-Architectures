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
import numpy as np

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

# Define the RNN Decoder (unchanged)
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        lstm_out, _ = self.lstm(inputs)
        outputs = self.fc(lstm_out[:, 1:])
        return outputs


def generate_caption(encoder, decoder, image_tensor, tokenizer, device, max_length=30):
    """
    Generate a caption for a single image
    """
    # Set models to eval mode
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        # Encode image
        image_tensor = image_tensor.unsqueeze(0).to(device)
        features = encoder(image_tensor)
        
        # Initialize caption generation
        caption = []
        input_word = torch.tensor([[tokenizer.cls_token_id]]).to(device)  # Start token
        
        # Generate caption word by word
        for _ in range(max_length):
            embeddings = decoder.embedding(input_word)
            if len(caption) == 0:
                lstm_input = torch.cat((features.unsqueeze(1), embeddings), 1)
            else:
                lstm_input = embeddings
                
            lstm_out, hidden = decoder.lstm(lstm_input)
            outputs = decoder.fc(lstm_out)
            
            # Get the most likely next word
            predicted = outputs.argmax(2)
            predicted_token_id = predicted[0, -1].item()
            
            # Stop if we predict the end token
            if predicted_token_id == tokenizer.sep_token_id:
                break
                
            caption.append(predicted_token_id)
            input_word = predicted
            
    # Convert token IDs to words
    caption = tokenizer.decode(caption, skip_special_tokens=True)
    return caption

def visualize_prediction(image_tensor, actual_caption, predicted_caption):
    """
    Display the image with actual and predicted captions
    """
    # Convert tensor to image
    image = image_tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title('Generated vs Actual Caption', pad=20)
    
    # Add captions as text below the image
    plt.figtext(0.1, 0.05, f'Actual: {actual_caption}', wrap=True, fontsize=10)
    plt.figtext(0.1, 0.02, f'Predicted: {predicted_caption}', wrap=True, fontsize=10)
    plt.show()

def test_model(checkpoint_path, test_coco_json, test_img_dir, num_test_samples=5):
    """
    Test the model on a few random images from the test set
    """
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models and load checkpoint
    embed_size = 256
    hidden_size = 512
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size
    
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    test_coco = COCO(test_coco_json)
    test_image_ids = test_coco.getImgIds()
    
    # Randomly sample images
    sample_ids = random.sample(test_image_ids, num_test_samples)
    
    # Test on sampled images
    for img_id in sample_ids:
        # Load and preprocess image
        img_info = test_coco.imgs[img_id]
        image_path = f"{test_img_dir}/{img_info['file_name']}"
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        
        # Get actual caption
        ann_ids = test_coco.getAnnIds(imgIds=img_id)
        anns = test_coco.loadAnns(ann_ids)
        actual_caption = anns[0]['caption']  # Get first caption
        
        # Generate caption
        predicted_caption = generate_caption(encoder, decoder, image_tensor, tokenizer, device)
        
        # Visualize results
        visualize_prediction(image_tensor, actual_caption, predicted_caption)

def calculate_metrics(checkpoint_path, test_coco_json, test_img_dir, num_test_samples=100):
    """
    Calculate BLEU score and other metrics on test set
    """
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.tokenize import word_tokenize
    import nltk
    try:
        nltk.download('punkt')
    except:
        pass
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models and load checkpoint
    embed_size = 256
    hidden_size = 512
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size
    
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    test_coco = COCO(test_coco_json)
    test_image_ids = test_coco.getImgIds()
    sample_ids = random.sample(test_image_ids, num_test_samples)
    
    # Calculate metrics
    bleu_scores = []
    
    for img_id in tqdm(sample_ids, desc="Calculating metrics"):
        # Load and preprocess image
        img_info = test_coco.imgs[img_id]
        image_path = f"{test_img_dir}/{img_info['file_name']}"
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        
        # Get actual caption
        ann_ids = test_coco.getAnnIds(imgIds=img_id)
        anns = test_coco.loadAnns(ann_ids)
        actual_caption = anns[0]['caption']
        
        # Generate caption
        predicted_caption = generate_caption(encoder, decoder, image_tensor, tokenizer, device)
        
        # Calculate BLEU score
        reference = word_tokenize(actual_caption.lower())
        candidate = word_tokenize(predicted_caption.lower())
        bleu_score = sentence_bleu([reference], candidate)
        bleu_scores.append(bleu_score)
    
    # Print metrics
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"\nMetrics on {num_test_samples} test samples:")
    print(f"Average BLEU score: {avg_bleu:.4f}")
    
    return avg_bleu

if __name__ == '__main__':
    # Paths
    checkpoint_path = 'checkpoints/checkpoint_epoch_5.pth'  # Adjust to your latest checkpoint
    test_coco_json = 'coco_dataset/annotations/captions_val2017.json'  # Validation set annotations
    test_img_dir = 'coco_dataset/val2017'  # Validation set images
    
    # Test the model
    print("Generating sample predictions...")
    test_model(checkpoint_path, test_coco_json, test_img_dir, num_test_samples=5)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    calculate_metrics(checkpoint_path, test_coco_json, test_img_dir, num_test_samples=100)