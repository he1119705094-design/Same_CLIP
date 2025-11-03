import os
import csv
import argparse

def create_csv_from_folder(base_folder, output_csv, dir=''):
    data = []

    for root, dirs, files in os.walk(base_folder):
        for file in sorted(files)[:3000]:
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.webp') or file.endswith('.jpeg')or file.endswith('.JPEG'):  # Assuming we're only interested in PNG files
                file_path = os.path.join(root, file)
                file_type = file_path.split(os.sep)[1]  # Extracting the type from the folder structure
                data.append((file_path, dir))
        if len(data) > 3000:
            break

    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['filename', 'typ'])
        csvwriter.writerows(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a CSV file from images in a folder.")
    parser.add_argument("base_folder", type=str, help="Path to the base folder containing images.")
    parser.add_argument("output_csv", type=str, help="Output CSV file path.")
    parser.add_argument("--dir", type=str, default='', help="Optional type label for the images.")

    args = parser.parse_args()
    create_csv_from_folder(args.base_folder, args.output_csv, dir=args.dir)








