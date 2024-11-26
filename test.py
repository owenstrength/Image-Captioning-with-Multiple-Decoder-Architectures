import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from pycocotools.coco import COCO
from pycocoevalcap.cider.cider import Cider
from PIL import Image
from tqdm import tqdm
from transformers import BertTokenizer
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk
from models.EncoderCNN import EncoderCNN
from models.DecoderModels import DecoderRNN, DecoderGRU, DecoderLSTM, DecoderLSTMAttention

nltk.download('punkt_tab')

def generate_caption(encoder, decoder, image_tensor, tokenizer, device):
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

def visualize_prediction(image, name, actual_caption, predicted_caption):
    """
    Save the image and captions in a matplotlib plot
    """
    # Create the figure and display
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Generated vs Actual Caption for {name}', pad=20, fontsize=16, fontweight='bold', loc='center')
    
    # Add captions
    plt.figtext(0.5, 0.05, f'Actual: {actual_caption}', wrap=True, fontsize=12, color='green', fontweight='bold', ha='center')
    plt.figtext(0.5, 0.02, f'Predicted: {predicted_caption}', wrap=True, fontsize=12, color='blue', fontweight='bold', ha='center')
    
    filename = f"./results/prediction_{name}_{random.randint(0, 10000)}.png"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # Close the figure to free memory
    
    print(f"Saved prediction to: {filename}")

def calculate_metrics(encoder, decoder, name, tokenizer, test_coco_json, test_img_dir, num_test_samples=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    CIDEr_scores = []
    METEOR_scores = []
    
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
            CIDEr = Cider().compute_score({img_id: [' '.join(reference_captions)]}, {img_id: [' '.join(predicted_caption)]})[0]
            meteor = meteor_score(references, candidate)
            
            bleu_1_scores.append(bleu_1)
            bleu_2_scores.append(bleu_2)
            bleu_3_scores.append(bleu_3)
            bleu_4_scores.append(bleu_4)
            CIDEr_scores.append(CIDEr)
            METEOR_scores.append(meteor)
            
            # Use BLEU-4 as the main score
            bleu_scores.append(bleu_4)
            
            # Print some examples
            if len(bleu_scores) % 100 == 0:
                print("\nExample caption:")
                print(f"Predicted: {predicted_caption}")
                print(f"Reference: {reference_captions[0]}")
                print(f"BLEU-4: {bleu_4:.4f}")
                print(f"CIDEr: {CIDEr}")
                print(f"METEOR: {meteor:.4f}")
                visualize_prediction(image, name, reference_captions[0], predicted_caption)
                
        except Exception as e:
            print(f"Error processing image {img_id}: {str(e)}")
            continue

    avg_bleu_1 = sum(bleu_1_scores) / len(bleu_1_scores)
    avg_bleu_2 = sum(bleu_2_scores) / len(bleu_2_scores)
    avg_bleu_3 = sum(bleu_3_scores) / len(bleu_3_scores)
    avg_bleu_4 = sum(bleu_4_scores) / len(bleu_4_scores)
    avg_CIDEr = sum(CIDEr_scores) / len(CIDEr_scores)
    avg_METEOR = sum(METEOR_scores) / len(METEOR_scores)
    
    print(f"\nMetrics on {len(bleu_scores)} test samples for {name}:")
    print(f"Average BLEU-1 score: {avg_bleu_1:.4f}")
    print(f"Average BLEU-2 score: {avg_bleu_2:.4f}")
    print(f"Average BLEU-3 score: {avg_bleu_3:.4f}")
    print(f"Average BLEU-4 score: {avg_bleu_4:.4f}")
    print(f"Average CIDEr score: {avg_CIDEr:.4f}")
    print(f"Average METEOR score: {avg_METEOR:.4f}")

if __name__ == '__main__':
    encoder_path = 'checkpoints/encoder-50.pkl'  # Path to encoder weights
    decoder_path = 'checkpoints/decoder-50.pkl'  # Path to decoder weights
    test_coco_json = 'coco_dataset/annotations/captions_val2017.json'
    test_img_dir = 'coco_dataset/val2017'

    try:
        nltk.download('punkt')
        nltk.download('wordnet') # For METEOR
    except:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embed_size = 256
    hidden_size = 512
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size

    paths = [
        ('rnn', 'models_checkpoints/encoder-rnn-5.pkl', 'models_checkpoints/decoder-rnn-5.pkl', DecoderRNN(embed_size, hidden_size, vocab_size).to(device)),
        ('gru', 'models_checkpoints/encoder-gru-5.pkl', 'models_checkpoints/decoder-gru-5.pkl', DecoderGRU(embed_size, hidden_size, vocab_size).to(device)),
        ('lstm', 'models_checkpoints/encoder-lstm-5.pkl', 'models_checkpoints/decoder-lstm-5.pkl', DecoderLSTM(embed_size, hidden_size, vocab_size).to(device)),
        ('lstm_attention', 'models_checkpoints/encoder-lstm_attention-5.pkl', 'models_checkpoints/decoder-lstm_attention-5.pkl', DecoderLSTMAttention(embed_size, hidden_size, vocab_size).to(device))
             ]

    for name, encoder_path, decoder_path, decoder in paths:
        print(f"Using encoder: {encoder_path}")
        print(f"Using decoder: {decoder_path}")

        encoder_weights = torch.load(encoder_path)
        decoder_weights = torch.load(decoder_path)
        
        encoder = EncoderCNN(embed_size).to(device)
            
        encoder.load_state_dict(encoder_weights)
        decoder.load_state_dict(decoder_weights)
    
        print("\nCalculating metrics...")
        calculate_metrics(encoder, decoder, name, tokenizer, test_coco_json, test_img_dir, num_test_samples=1000)