"""
图像描述生成模块
使用BLIP或InstructBLIP为图像生成文本描述，用于多模态特征提取
"""

import torch
import torch.nn as nn
from PIL import Image
import os
import json
from tqdm import tqdm
import argparse

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False
    print("Warning: BLIP/InstructBLIP not available. Install with: pip install transformers")

class ImageCaptioner:
    """图像描述生成器"""
    
    def __init__(self, model_type="blip", device="cuda"):
        self.device = device
        self.model_type = model_type
        
        if not BLIP_AVAILABLE:
            raise ImportError("BLIP/InstructBLIP not available. Please install transformers.")
        
        if model_type == "blip":
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        elif model_type == "instructblip":
            self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
            self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model.to(device)
        self.model.eval()
    
    def generate_caption(self, image, prompt=None):
        """为单张图像生成描述"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        if self.model_type == "blip":
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_length=50, num_beams=5)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
        elif self.model_type == "instructblip":
            if prompt is None:
                prompt = "Describe this image in detail."
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_length=50, num_beams=5)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        return caption
    
    def generate_captions_batch(self, image_paths, output_file=None, prompt=None):
        """批量生成图像描述"""
        captions = {}
        
        for img_path in tqdm(image_paths, desc="Generating captions"):
            try:
                caption = self.generate_caption(img_path, prompt)
                captions[img_path] = caption
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                captions[img_path] = ""
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(captions, f, ensure_ascii=False, indent=2)
        
        return captions

def add_captioning_arguments(parser):
    """添加图像描述相关参数"""
    parser.add_argument("--enable_captioning", action="store_true", 
                       help="Enable automatic image captioning")
    parser.add_argument("--caption_model", type=str, default="blip", 
                       choices=["blip", "instructblip"],
                       help="Captioning model to use")
    parser.add_argument("--caption_prompt", type=str, default="Describe this image in detail.",
                       help="Prompt for InstructBLIP")
    parser.add_argument("--caption_file", type=str, default="captions.json",
                       help="Path to save/load captions")
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_captioning_arguments(parser)
    args = parser.parse_args()
    
    # 示例用法
    captioner = ImageCaptioner(model_type=args.caption_model)
    
    # 为数据集生成描述
    dataset_path = "path/to/dataset"
    image_paths = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    captions = captioner.generate_captions_batch(
        image_paths, 
        output_file=args.caption_file,
        prompt=args.caption_prompt
    )
    
    print(f"Generated {len(captions)} captions")
