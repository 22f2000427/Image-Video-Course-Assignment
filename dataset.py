import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class SmileDataset(Dataset):
    def __init__(self, img_dir, attr_path, transform=None, limit=None):
        self.img_dir = img_dir
        self.transform = transform

        
        with open(attr_path, 'r') as f:
            lines = f.readlines()
            header_line = lines[0].strip().split()
            data_lines = lines[1:]

        
        data = []
        for line in data_lines:
            parts = line.strip().split()
            image_id = parts[0]
            attributes = list(map(int, parts[1:]))
            data.append([image_id] + attributes)

        
        self.attrs = pd.DataFrame(data, columns=['image_id'] + header_line)

        print("Successfully parsed attribute file.")
        print(f"Attribute columns: {self.attrs.columns.tolist()}")
        print(f"Sample rows:\n{self.attrs.head()}")

        
        if limit:
            self.attrs = self.attrs.iloc[:limit]

    def __len__(self):
        return len(self.attrs)

    def __getitem__(self, idx):
        row = self.attrs.iloc[idx]
        image_id = row['image_id']
        img_path = os.path.join(self.img_dir, image_id)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        
        label = 1 if int(row['Smiling']) == 1 else 0
        return image, label


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


def unicornLoader(batch_size=64, limit=10000):
    dataset = SmileDataset(
        img_dir="celebA/img_align_celeba", 
        attr_path="celebA/list_attr_celeba.txt",
        transform=transform,
        limit=limit
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
