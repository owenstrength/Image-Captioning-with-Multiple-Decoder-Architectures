

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
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.rouge.rouge import Rouge
from models.EncoderCNN import EncoderCNN
from models.DecoderRNN import DecoderRNN

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

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
        
        # Use decoder's sample method
        sampled_ids = decoder.sample(features.unsqueeze(1))
        
        # Convert indices to words
        caption = tokenizer.decode(sampled_ids, skip_special_tokens=True)
        
    return caption

def visualize_prediction(image, actual_caption, predicted_caption):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title('Generated vs Actual Caption', pad=20, fontsize=16, fontweight='bold', loc='center')
    
    plt.figtext(0.5, 0.05, f'Actual: {actual_caption}', wrap=True, fontsize=12, color='green', fontweight='bold', ha='center')
    plt.figtext(0.5, 0.02, f'Predicted: {predicted_caption}', wrap=True, fontsize=12, color='blue', fontweight='bold', ha='center')
    
    filename = f"./results/prediction_{random.randint(0, 10000)}.png"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Saved prediction to: {filename}")

def calculate_meteor_score(references, hypothesis):
    """
    Calculate METEOR score for a single prediction
    """
    return meteor_score(references, hypothesis)

def prepare_for_cider(references, hypotheses):
    """
    Prepare data in the format required by CIDEr scorer
    """
    refs = {}
    hyps = {}
    for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
        refs[i] = [' '.join(r) for r in ref]
        hyps[i] = [' '.join(hyp)]
    return refs, hyps

def test_model(encoder_weights, decoder_weights, test_coco_json, test_img_dir, num_test_samples=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    
    # Model initialization
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
    
    # Initialize scorers
    cider_scorer = Cider()
    rouge_scorer = Rouge()
    
    # Lists to store all references and hypotheses
    all_references = []
    all_hypotheses = []
    meteor_scores = []
    
    for img_id in tqdm(sample_ids, desc="Processing images"):
        try:
            # Process image
            img_info = test_coco.imgs[img_id]
            image_path = f"{test_img_dir}/{img_info['file_name']}"
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image)
            
            # Get reference captions
            ann_ids = test_coco.getAnnIds(imgIds=img_id)
            anns = test_coco.loadAnns(ann_ids)
            reference_captions = [word_tokenize(ann['caption'].lower()) for ann in anns]
            
            # Generate prediction
            predicted_caption = generate_caption(encoder, decoder, image_tensor, tokenizer, device)
            hypothesis = word_tokenize(predicted_caption.lower())
            
            # Calculate METEOR score for this sample
            meteor_scores.append(calculate_meteor_score(reference_captions, hypothesis))
            
            # Add to corpus collections
            all_references.append(reference_captions)
            all_hypotheses.append(hypothesis)
            
            # Print examples occasionally
            if len(all_hypotheses) % 10 == 0:
                print("\nExample caption:")
                print(f"Predicted: {predicted_caption}")
                print(f"Reference: {' '.join(reference_captions[0])}")
                visualize_prediction(image, ' '.join(reference_captions[0]), predicted_caption)
                
        except Exception as e:
            print(f"Error processing image {img_id}: {str(e)}")
            continue

    # Calculate BLEU scores
    bleu_1 = corpus_bleu(all_references, all_hypotheses, weights=(1.0, 0, 0, 0))
    bleu_2 = corpus_bleu(all_references, all_hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(all_references, all_hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = corpus_bleu(all_references, all_hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    # Calculate average METEOR score
    meteor_score_avg = np.mean(meteor_scores)
    
    # Calculate CIDEr and ROUGE scores
    refs, hyps = prepare_for_cider(all_references, all_hypotheses)
    cider_score, _ = cider_scorer.compute_score(refs, hyps)
    rouge_score, _ = rouge_scorer.compute_score(refs, hyps)
    
    # Print all metrics
    print(f"\nEvaluation metrics on {len(all_hypotheses)} test samples:")
    print(f"BLEU-1 score: {bleu_1:.4f}")
    print(f"BLEU-2 score: {bleu_2:.4f}")
    print(f"BLEU-3 score: {bleu_3:.4f}")
    print(f"BLEU-4 score: {bleu_4:.4f}")
    print(f"METEOR score: {meteor_score_avg:.4f}")
    print(f"CIDEr score: {cider_score:.4f}")
    print(f"ROUGE score: {rouge_score:.4f}")

    return {
        'bleu_1': bleu_1,
        'bleu_2': bleu_2,
        'bleu_3': bleu_3,
        'bleu_4': bleu_4,
        'meteor': meteor_score_avg,
        'cider': cider_score,
        'rouge': rouge_score,
        'num_samples': len(all_hypotheses)
    }

if __name__ == '__main__':
    encoder_path = 'checkpoints/GRU/encoder-10.pkl'
    decoder_path = 'checkpoints/GRU/decoder-10.pkl'
    test_coco_json = 'coco_dataset/annotations/captions_val2017.json'
    test_img_dir = 'coco_dataset/val2017'

    encoder_weights = torch.load(encoder_path)
    decoder_weights = torch.load(decoder_path)
    
    print("Generating sample predictions...")
    test_model(encoder_weights, decoder_weights, test_coco_json, test_img_dir, num_test_samples=5)
    
    print("\nCalculating metrics...")
    metrics = calculate_metrics(encoder_weights, decoder_weights, test_coco_json, test_img_dir, num_test_samples=100)