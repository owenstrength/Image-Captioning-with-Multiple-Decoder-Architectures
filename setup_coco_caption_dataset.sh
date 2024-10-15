#!/bin/bash

# Create a directory to store COCO dataset
mkdir -p coco_dataset
cd coco_dataset

# Download train2017 images (13GB)
echo "Downloading train2017 images..."
wget http://images.cocodataset.org/zips/train2017.zip
echo "Extracting train2017 images..."
unzip train2017.zip
rm train2017.zip

# Download val2017 images (6GB)
echo "Downloading val2017 images..."
wget http://images.cocodataset.org/zips/val2017.zip
echo "Extracting val2017 images..."
unzip val2017.zip
rm val2017.zip

# Download captions annotations for train and val
echo "Downloading captions annotations..."
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
echo "Extracting annotations..."
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip

# Install pycocotools if not installed
if ! python -c "import pycocotools" &> /dev/null; then
    echo "pycocotools not found, installing it..."
    pip install pycocotools
else
    echo "pycocotools is already installed."
fi

# Folder structure summary
echo "COCO dataset setup complete!"
echo "Folder structure:"
tree -L 2

