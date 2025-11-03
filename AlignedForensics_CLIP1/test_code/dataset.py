from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os

class UnlabeledImageDataset(Dataset):
    def __init__(self, folder, transform=None,split=None, return_names=False, count=100):
        self.folder = folder
        self.return_names = return_names
        self.transform = transform
        self.image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
        self.image_files=self.image_files[:count] 

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.return_names:
            return image, img_path
        return image
