import os
import numpy as np
import pandas as pd
from PIL import Image

# label number -> letter (no J=9 or Z=25 in this dataset)
letters = "ABCDEFGHIKLMNOPQRSTUVWXY"
label_map = {i: c for i, c in zip([x for x in range(25) if x != 9], letters)}

def convert(csv_path, out_dir):
    df = pd.read_csv(csv_path)
    for i, row in df.iterrows():
        letter = label_map[int(row["label"])]
        folder = os.path.join(out_dir, letter)
        os.makedirs(folder, exist_ok=True)
        pixels = row.drop("label").values.astype(np.uint8).reshape(28, 28)
        img = Image.fromarray(pixels).convert("RGB").resize((64, 64))
        img.save(os.path.join(folder, f"{i}.png"))

convert("sign_mnist_train.csv", "data/train")
convert("sign_mnist_test.csv", "data/val")
print("done")