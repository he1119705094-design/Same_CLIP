from PIL import Image, ImageFilter
import numpy as np
from recon_local import *
from tqdm import tqdm
import torch
import os
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import imageio
import random
import torchvision.transforms.functional as TF


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_random_mask(image_tensor, mask_ratio=0.5):
    """
    在图像中随机生成一个矩形 mask 区域。
    输入：image_tensor [C,H,W]，mask_ratio 表示mask占图像比例
    输出：mask Tensor，shape [1,H,W]，值为0/1
    """
    _, H, W = image_tensor.shape
    mask = torch.zeros((1, H, W), dtype=torch.float32)
    h = int(H * mask_ratio)
    w = int(W * mask_ratio)
    y1 = random.randint(0, H - h)
    x1 = random.randint(0, W - w)
    mask[:, y1:y1+h, x1:x1+w] = 1.0
    return mask

class UnlabeledImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder)
                            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]

def get_vae(repo_id):
    print(f"Loading VAE from {repo_id}...")
    vae = AutoencoderKL.from_pretrained(
        repo_id,
        subfolder="vqvae",
        cache_dir="weights",
        use_safetensors=False,        # ✅ 关闭 .safetensors
        local_files_only=True         # ✅ 只使用本地文件
    )
    print("✅ VAE 加载成功，模型结构:")
    print(vae)
    return vae

def save_images(images_tensor, output_folder, names):
    os.makedirs(output_folder, exist_ok=True)
    for idx, image_tensor in enumerate(images_tensor):
        name = os.path.basename(names[idx])
        name = name.replace('.jpg', '.png')  # 或根据实际情况处理
        if torch.is_tensor(image_tensor):
            image_pil = transforms.ToPILImage()(image_tensor)
        else:
            image_pil = image_tensor
        image_path = os.path.join(output_folder, name)
        image_pil.save(image_path)


def create_dataloader(image_folder, batch_size=32, shuffle=True, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 或其他统一尺寸
        transforms.ToTensor(),
    ])

    dataset = UnlabeledImageDataset(image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


# def recon(pipe, dataloader, ae, seed, args, tools):
#     for batch_idx, (images, names) in enumerate(dataloader):
#         if batch_idx == 0:
#             print(f"Batch {batch_idx + 1}:")
#             print(f" - Images shape: {images.shape}")
#         with torch.no_grad():
#             recons = reconstruct_simple(x=images.to(device), ae=ae, seed=seed, steps=args.steps, tools=tools)
#         save_images(images_tensor=recons, input_path=args.input_folder, names=names)
"""
下面这个部分是全局伪造的代码
def recon(pipe, dataloader, ae, seed, args, tools):
    for batch_idx, (images, names) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f" - Images shape: {images.shape}")
        with torch.no_grad():
            recons = reconstruct_simple(x=images.to(device), ae=ae, seed=seed, steps=args.steps, tools=tools)
        save_images(images_tensor=recons, output_folder=args.output_folder, names=names)
"""

"""
#recon_local函数是局部伪造的代码
def recon_local(pipe, dataloader, ae, seed, args, tools):
    for batch_idx, (images, names) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1} - Images shape: {images.shape}")
        with torch.no_grad():
            images = images.to(device)
            recons = reconstruct_simple(images, ae, seed, steps=args.steps, tools=tools)

            output_images = []
            for i in range(images.size(0)):
                mask = generate_random_mask(images[i], mask_ratio=0.4).to(device)
                composite = recons[i] * mask + images[i] * (1 - mask)
                output_images.append(composite)

            output_images = torch.stack(output_images)
            save_images(output_images, args.output_folder, names)
"""

#更改为可视化伪造局部区域的代码
def recon_local(pipe, dataloader, ae, seed, args, tools):
    mask_vis_dir = os.path.join(args.output_folder, 'mask_vis')
    os.makedirs(mask_vis_dir, exist_ok=True)

    for batch_idx, (images, names) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1} - Images shape: {images.shape}")
        with torch.no_grad():
            images = images.to(device)
            recons = reconstruct_simple(images, ae, seed, steps=args.steps, tools=tools)

            output_images = []
            for i in range(images.size(0)):
                mask = generate_random_mask(images[i], mask_ratio=0.5).to(device)
                composite = recons[i] * mask + images[i] * (1 - mask)
                output_images.append(composite)

                # ✅ 可视化保存 mask 区域叠加图
                img_pil = TF.to_pil_image(images[i].cpu())
                mask_np = mask[0].cpu().numpy()
                mask_img = Image.fromarray((mask_np * 255).astype('uint8')).resize(img_pil.size)

                # 红色遮罩叠加
                red_overlay = Image.new('RGB', img_pil.size, (255, 0, 0))
                img_with_mask = Image.blend(img_pil, red_overlay, alpha=0.5)
                img_with_mask.putalpha(mask_img)

                name = os.path.basename(names[i])
                name = name.replace('.jpg', '.png')
                img_with_mask.save(os.path.join(mask_vis_dir, name))

            output_images = torch.stack(output_images)
            save_images(output_images, args.output_folder, names)

# def main():
#     parser = argparse.ArgumentParser(description='Create a DataLoader from an image folder dataset.')
#     parser.add_argument('--input_folder', type=str, help='Path to the input folder containing images.')
#     parser.add_argument('--repo_id', type=str, help='Correct stable diffusion autoencoder')
#     parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader.')
#     parser.add_argument('--shuffle', action='store_true', help='Whether to shuffle the data in the DataLoader.')
#     parser.add_argument('--steps', type=int, default=None, help='Number of backward steps in DDIM inversion')
#     parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for DataLoader.')
#     args = parser.parse_args()
#     seed = 42
#     dataloader = create_dataloader(args.input_folder, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
#
#     ae = get_vae(repo_id=args.repo_id).to(device)
#     tools = None
#     recon(pipe=None, dataloader=dataloader, ae=ae, seed=seed, args=args, tools=tools)

def main():
    parser = argparse.ArgumentParser(description='Create a DataLoader from an image folder dataset.')
    parser.add_argument('--input_folder', type=str, help='Path to the input folder containing images.')
    parser.add_argument('--output_folder', type=str, help='Path to save the reconstructed images.')
    parser.add_argument('--repo_id', type=str, help='Correct stable diffusion autoencoder')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader.')
    parser.add_argument('--shuffle', action='store_true', help='Whether to shuffle the data in the DataLoader.')
    parser.add_argument('--steps', type=int, default=None, help='Number of backward steps in DDIM inversion')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for DataLoader.')
    args = parser.parse_args()

    seed = 42
    dataloader = create_dataloader(args.input_folder, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
    ae = get_vae(repo_id=args.repo_id).to(device)
    tools = None
    # 原来的调用命令换为新的调用命令了 recon(pipe=None, dataloader=dataloader, ae=ae, seed=seed, args=args, tools=tools)
    recon_local(pipe=None, dataloader=dataloader, ae=ae, seed=seed, args=args, tools=tools)

if __name__ == "__main__":
    main()