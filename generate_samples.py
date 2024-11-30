import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import random
from pycocotools.coco import COCO
from models.EncoderCNN import EncoderCNN
from models.DecoderModels import DecoderRNN, DecoderGRU, DecoderLSTM, DecoderLSTMAttention
import os
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib.gridspec as gridspec

def setup_models(device, embed_size=256, hidden_size=512):
    """Setup all models with their respective weights"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size
    
    model_configs = {
        'rnn': DecoderRNN(embed_size, hidden_size, vocab_size),
        'gru': DecoderGRU(embed_size, hidden_size, vocab_size),
        'lstm': DecoderLSTM(embed_size, hidden_size, vocab_size),
    }
    
    models = {}
    encoders = {}
    
    # Load weights for each model-encoder pair
    for model_name, decoder in model_configs.items():
        decoder = decoder.to(device)
        encoder = EncoderCNN(embed_size).to(device)
        
        decoder_path = f'models_checkpoints/decoder-{model_name}-5.pkl'
        encoder_path = f'models_checkpoints/encoder-{model_name}-5.pkl'
        
        if os.path.exists(decoder_path) and os.path.exists(encoder_path):
            decoder.load_state_dict(torch.load(decoder_path, map_location=device))
            encoder.load_state_dict(torch.load(encoder_path, map_location=device))
            
            models[model_name] = decoder
            encoders[model_name] = encoder
        else:
            print(f"Warning: Could not find weights for {model_name}")
    
    return encoders, models, tokenizer

def generate_caption(encoder, decoder, image_tensor, tokenizer, device):
    """Generate caption for an image"""
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        features = encoder(image_tensor)
        sampled_ids = decoder.sample(features.unsqueeze(1))
        caption = tokenizer.decode(sampled_ids, skip_special_tokens=True)
    
    return caption


def save_comparison(image, references, predictions, save_path, meteor_scores):
    """Create a visually appealing comparison with better layout"""
    # Set the style
    sns.set_palette("husl")
    
    # Create figure with custom grid
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    # Left plot - Image
    ax0 = plt.subplot(gs[0])
    ax0.imshow(image)
    ax0.axis('off')
    ax0.set_title('Input Image', pad=20, fontsize=16, fontweight='bold')
    
    # Right plot - Captions
    ax1 = plt.subplot(gs[1])
    ax1.axis('off')
    
    # Define colors for different sections
    reference_color = '#2E5090'
    model_colors = {
        'rnn': '#FF6B6B',
        'gru': '#4ECDC4',
        'lstm': '#45B7D1',
        'lstm_attention': '#96CEB4'
    }
    
    # Add reference captions
    y_pos = 0.95
    ax1.text(0, 1.0, 'Ground Truth Captions:', fontsize=18, fontweight='bold', color=reference_color)
    for i, ref in enumerate(references, 1):
        ax1.text(0.05, y_pos, f"{i}. {ref}", fontsize=16, 
                bbox=dict(facecolor='white', edgecolor=reference_color, alpha=0.1, pad=5))
        y_pos -= 0.07
    
    # Add separator
    y_pos -= 0.03
    ax1.axhline(y=y_pos, color='gray', linestyle='--', xmin=0.05, xmax=0.95, alpha=0.3)
    y_pos -= 0.05
    
    # Add model predictions
    ax1.text(0, y_pos, 'Model Predictions:', fontsize=18, fontweight='bold')
    y_pos -= 0.05
    
    for model_name, pred in predictions.items():
        display_name = model_name.upper().replace('_', '+')
        color = model_colors.get(model_name, '#666666')
        
        # Create a box for each prediction
        bbox_props = dict(boxstyle="round,pad=0.5", facecolor='white', 
                         edgecolor=color, alpha=0.1)
        
        # Add model name and BLEU score
        ax1.text(0.05, y_pos, f"{display_name} (METEOR: {meteor_scores[model_name]:.2f})", 
                fontsize=16, color=color, fontweight='bold')
        y_pos -= 0.05
        
        # Add prediction text
        ax1.text(0.1, y_pos, pred, fontsize=15, 
                bbox=bbox_props, wrap=True)
        y_pos -= 0.08
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_samples(num_samples=10):
    """Generate sample outputs from all models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup models
    encoders, decoders, tokenizer = setup_models(device)
    
    if not encoders or not decoders:
        print("No models loaded. Please check model checkpoint paths.")
        return
    
    # Setup data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load COCO validation set
    coco = COCO('coco_dataset/annotations/captions_val2017.json')
    image_ids = coco.getImgIds()
    
    # Create output directory
    os.makedirs('report_samples', exist_ok=True)
    
    # Generate samples
    for i in range(num_samples):
        try:
            # Get random image
            img_id = random.choice(image_ids)
            img_info = coco.imgs[img_id]
            image_path = f"coco_dataset/val2017/{img_info['file_name']}"
            
            # Load and transform image
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image)
            
            # Get ALL reference captions
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            references = [ann['caption'] for ann in anns]
            reference_tokens = [word_tokenize(ref.lower()) for ref in references]
            
            # Generate predictions from all models
            predictions = {}
            bleu_scores = {}
            meteor_scores = {}
            
            for model_name in decoders.keys():
                pred = generate_caption(
                    encoders[model_name],
                    decoders[model_name],
                    image_tensor,
                    tokenizer,
                    device
                )
                predictions[model_name] = pred
                
                # Calculate BLEU score using all references
                prediction_tokens = word_tokenize(pred.lower())
                meteor = meteor_score(reference_tokens, prediction_tokens)
                max_bleu = max([sentence_bleu([ref], prediction_tokens) for ref in reference_tokens])
                bleu_scores[model_name] = max_bleu
                meteor_scores[model_name] = meteor
            
            # Save results
            save_path = f'report_samples/sample_{i+1}.png'
            save_comparison(image, references, predictions, save_path, meteor_scores)
            print(f"Generated sample {i+1}/{num_samples}")
            
            # Print the captions for this sample
            print(f"\nSample {i+1}:")
            print("References:")
            for j, ref in enumerate(references, 1):
                print(f"{j}. {ref}")
            print("\nPredictions:")
            for model_name, pred in predictions.items():
                print(f"{model_name.upper()}: {pred} (BLEU: {bleu_scores[model_name]:.2f})")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error generating sample {i+1}: {str(e)}")
            continue



if __name__ == '__main__':
    generate_samples(num_samples=20)  # Generate 20 samples
    print("Sample generation complete!")