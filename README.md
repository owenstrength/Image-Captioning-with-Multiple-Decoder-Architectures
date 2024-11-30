# Image Captioning with Multiple Decoder Architectures

This project implements an image captioning system using various decoder architectures (RNN, GRU, and LSTM) on the COCO 2017 dataset. The system uses a pre-trained ResNet-50 as the encoder and different recurrent architectures for caption generation.

## Project Structure
```
├── coco_dataset/
│   ├── annotations/
│   │   ├── captions_train2017.json
│   │   └── captions_val2017.json
│   ├── train2017/
│   └── val2017/
├── models/
│   ├── EncoderCNN.py
│   └── DecoderModels.py
├── train.py
├── test.py
└── requirements.txt
```

## Setup and Dependencies

### Requirements
```
torch
torchvision
transformers
pycocotools
nltk
pillow
tqdm
numpy
matplotlib
pycocoevalcap
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Dataset
The project uses the COCO 2017 dataset. Download it from [COCO website](https://cocodataset.org/#download) and place it in the `coco_dataset` directory.

We have also created a shell file that does this for you. Run `./setup_coco_caption_dataset.sh` to download the dataset.

## Files Description

### 1. models/EncoderCNN.py
Implements the encoder using a pre-trained ResNet-50 model.

**Classes:**
- `EncoderCNN`: Extracts image features using ResNet-50
  - Input: Images (batch_size, 3, 224, 224)
  - Output: Image features (batch_size, embed_size)

### 2. models/DecoderModels.py
Contains different decoder architectures for caption generation.

**Classes:**
- `DecoderRNN`: Basic RNN decoder
- `DecoderGRU`: GRU-based decoder
- `DecoderLSTM`: LSTM-based decoder

Each decoder class implements:
- `forward()`: Training forward pass
- `sample()`: Caption generation for inference

### 3. train.py
Main training script for the image captioning models.

**Key Components:**
- `CocoDataset`: Custom dataset class for COCO
- `train_model()`: Training loop implementation
  - Inputs:
    - encoder: CNN encoder model
    - decoder: RNN decoder model
    - data_loader: Training data loader
    - num_epochs: Number of training epochs
  - Outputs:
    - Saved model checkpoints

### 4. test.py
Evaluation script for trained models.

**Key Functions:**
- `generate_caption()`: Generates caption for a single image
- `calculate_metrics()`: Computes evaluation metrics
  - BLEU-1,2,3,4
  - METEOR
  - CIDEr
  - ROUGE
- `visualize_prediction()`: Creates visualization of predictions

## Model Architectures

1. **Encoder**:
   - Pre-trained ResNet-50
   - Feature dimension: 2048 → embed_size

2. **Decoders**:
   - RNN: Basic recurrent neural network
   - GRU: Gated Recurrent Unit
   - LSTM: Long Short-Term Memory

## Usage

### Training
```bash
python train.py
```
This will train all decoder variants sequentially.

### Testing
```bash
python test.py
```
This evaluates all trained models and computes metrics.

## Model Checkpoints
Checkpoints are saved in the following format:
- Encoder: `checkpoints/encoder-{model_type}-{epoch}.pkl`
- Decoder: `checkpoints/decoder-{model_type}-{epoch}.pkl`

where `model_type` is one of: ['rnn', 'gru', 'lstm']

## Results
Results are saved in:
- Model predictions: `results/prediction_{name}_{random_id}.png`
- Metrics are printed for each model showing:
  - BLEU scores (1-4)
  - METEOR score
  - CIDEr score
  - ROUGE score

## Parameters
- Embedding size: 256
- Hidden size: 512
- Batch size: 128
- Learning rate: 0.001
- Number of epochs: 5
