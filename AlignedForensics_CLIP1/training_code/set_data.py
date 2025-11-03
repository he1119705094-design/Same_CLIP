import os
import shutil

root = r"D:\Model\AlignedForensics-master\datasets\train"
output_real = os.path.join(root, "0_real")
output_fake = os.path.join(root, "1_fake")

os.makedirs(output_real, exist_ok=True)
os.makedirs(output_fake, exist_ok=True)

for cls in ["airplane", "bicycle", "bird"]:
    for label in ["0_real", "1_fake"]:
        src_dir = os.path.join(root, cls, label)
        dst_dir = output_real if label == "0_real" else output_fake

        if os.path.exists(src_dir):
            for fname in os.listdir(src_dir):
                src_path = os.path.join(src_dir, fname)
                dst_path = os.path.join(dst_dir, f"{cls}_{fname}")
                shutil.copy2(src_path, dst_path)