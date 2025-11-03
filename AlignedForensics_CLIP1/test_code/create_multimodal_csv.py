"""
多模态测试CSV生成脚本
按照原始create_csv.py的格式，支持多模态数据
"""

import os
import csv
import argparse

def create_multimodal_csv_from_folder(base_folder, output_csv, dir=''):
    """创建多模态测试CSV文件"""
    data = []

    for root, dirs, files in os.walk(base_folder):
        for file in sorted(files)[:3000]:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                file_path = os.path.join(root, file)
                file_type = file_path.split(os.sep)[1]  # 从文件夹结构提取类型
                data.append((file_path, dir))
        if len(data) > 3000:
            break

    if not data:
        print(f"Warning: No valid image files found in {base_folder}")
        return
        # 自动创建输出目录
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['filename', 'typ'])
        csvwriter.writerows(data)
    print(f"CSV generated successfully: {output_csv}, total {len(data)} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a CSV file from images in a folder for multimodal testing.")
    parser.add_argument("base_folder", type=str, help="Path to the base folder containing images.")
    parser.add_argument("output_csv", type=str, help="Output CSV file path.")
    parser.add_argument("--dir", type=str, default='', help="Optional type label for the images.")

    args = parser.parse_args()
    create_multimodal_csv_from_folder(args.base_folder, args.output_csv, dir=args.dir)

