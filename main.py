import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from PIL import Image
import os

# Step 1: Load COCO dataset
class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.anns.keys())
        self.transform = transform

    def __getitem__(self, index):
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, img_path)).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        
        return image, caption

    def __len__(self):
        return len(self.ids)


# Step 2: Define the Image Encoder (ResNet)
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # Remove the last fully connected layer
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        features = self.relu(features)
        features = self.dropout(features)
        return features


# Step 3: Define the Caption Decoder (RNN)
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, features, states=None):
        "Generate captions for given image features (Greedy decoding)."
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids


# Step 4: Set Hyperparameters and Train the Model
def main():
    # Paths to your dataset (replace with your paths)
    train_image_dir = 'path/to/train2017'
    train_captions_file = 'path/to/annotations/captions_train2017.json'
    
    # Hyperparameters
    embed_size = 256
    hidden_size = 512
    vocab_size = 5000  # Adjust this according to your tokenizer
    num_layers = 1
    learning_rate = 0.001
    num_epochs = 5
    batch_size = 32

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Load dataset
    dataset = CocoDataset(root=train_image_dir, annFile=train_captions_file, transform=transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Build models
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.fc.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # Training the models
    for epoch in range(num_epochs):
        for i, (images, captions) in enumerate(data_loader):
            images = images
            captions = captions

            # Forward pass
            features = encoder(images)
            outputs = decoder(features, captions[:, :-1])
            loss = criterion(outputs.view(-1, vocab_size), captions[:, 1:].reshape(-1))

            # Backward pass and optimization
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Log info
            if i % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(data_loader)}], Loss: {loss.item():.4f}')

    print("Training complete.")


if __name__ == "__main__":
    main()
