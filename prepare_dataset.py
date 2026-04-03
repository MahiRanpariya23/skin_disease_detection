import os
import pandas as pd
import shutil

df = pd.read_csv("HAM10000_metadata.csv")

src_dirs = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
dest_dir = "dataset"

# create folders
for label in df['dx'].unique():
    os.makedirs(os.path.join(dest_dir, label), exist_ok=True)

# move images
for _, row in df.iterrows():
    img_name = row['image_id'] + ".jpg"
    label = row['dx']

    for src in src_dirs:
        src_path = os.path.join(src, img_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(dest_dir, label, img_name))
            break

print("Dataset ready!")