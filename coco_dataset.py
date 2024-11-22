from pycocotools.coco import COCO
from PIL import Image
from torch.utils.data import Dataset
import random

class CocoDataset(Dataset):
    def __init__(self, coco_json, tokenizer, transform=None, img_dir='', subset_fraction=0.001):
        self.coco = COCO(coco_json)
        all_image_ids = self.coco.getImgIds()
        
        # Calculate number of images for subset
        subset_size = max(1, int(len(all_image_ids) * subset_fraction))
        self.image_ids = random.sample(all_image_ids, subset_size)
        
        print(f"Using {len(self.image_ids)} images out of {len(all_image_ids)} total images")
        
        self.transform = transform
        self.tokenizer = tokenizer
        self.img_dir = img_dir
        
        # Create caption lengths list
        self.caption_lengths = {}
        for img_id in self.image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            self.caption_lengths[img_id] = len(anns[0]['caption'].split())

    def get_train_indices(self):
        sel_length = random.choice(list(self.caption_lengths.values()))
        return [i for i, v in enumerate(self.image_ids) 
                if self.caption_lengths[v] == sel_length]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        
        image_path = f"{self.img_dir}/{img_info['file_name']}"
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        caption = anns[0]['caption']
        
        caption_tokens = self.tokenizer(caption, padding='max_length', max_length=128, 
                                      return_tensors='pt', truncation=True)
        return image, caption_tokens['input_ids'].squeeze(0)
