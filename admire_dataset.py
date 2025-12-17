import os
import glob
import ast
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class AdMIReReader(Dataset):
    def __init__(self, data_root_path, split="Train", mode="qwen"):
        self.mode = mode
        
        # 1. AUTO-FIND TSV
        print(f"🕵️ Scanning {data_root_path} for {split} TSV...")
        tsv_candidates = glob.glob(os.path.join(data_root_path, "**", f"*{split}*.tsv"), recursive=True)
        if not tsv_candidates:
             tsv_candidates = glob.glob(os.path.join(data_root_path, "**", "*.tsv"), recursive=True)
        
        if not tsv_candidates:
            raise FileNotFoundError(f"❌ TSV file not found in {data_root_path}")
            
        self.tsv_path = tsv_candidates[0]
        print(f"✅ Loaded TSV: {self.tsv_path}")
        self.df = pd.read_csv(self.tsv_path, sep='\t')

        # 2. AUTO-DETECT IMAGE ROOT
        # We search for the first image to locate the real root folder
        first_img_name = self.df.iloc[0]['image1_name']
        found_images = glob.glob(os.path.join(data_root_path, "**", first_img_name), recursive=True)
        
        if not found_images:
            tsv_dir = os.path.dirname(self.tsv_path)
            found_images = glob.glob(os.path.join(tsv_dir, "**", first_img_name), recursive=True)

        if found_images:
            full_path = found_images[0]
            compound_folder = str(self.df.iloc[0]['compound'])
            
            # Check if image is inside a compound folder
            if compound_folder in full_path:
                split_point = full_path.find(compound_folder)
                self.image_root = full_path[:split_point]
            else:
                self.image_root = os.path.dirname(full_path)
            print(f"✅ Images located at: {self.image_root}")
        else:
            print("⚠️ WARNING: Could not auto-detect root. Using provided path.")
            self.image_root = data_root_path

        # 3. TRANSFORM (For Fusion)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        idiom_folder_raw = str(row['compound'])
        
        # --- FIX: HANDLE APOSTROPHES (devil's -> devil_s) ---
        # 1. Check if the folder exists exactly as written
        path_exact = os.path.join(self.image_root, idiom_folder_raw)
        
        # 2. Check if it exists with apostrophe replaced by underscore
        idiom_folder_sanitized = idiom_folder_raw.replace("'", "_")
        path_sanitized = os.path.join(self.image_root, idiom_folder_sanitized)
        
        if os.path.exists(path_exact):
            final_folder = idiom_folder_raw
        elif os.path.exists(path_sanitized):
            final_folder = idiom_folder_sanitized
        else:
            final_folder = idiom_folder_raw # Fallback (will fail later if missing)

        img_paths = []
        for i in range(1, 6):
            fname = row[f'image{i}_name']
            
            # Try with detected folder (Standard)
            path_a = os.path.join(self.image_root, final_folder, fname)
            # Try flat (Fallback)
            path_b = os.path.join(self.image_root, fname)
            
            if os.path.exists(path_a): final = path_a
            elif os.path.exists(path_b): final = path_b
            else: final = "MISSING"
            
            img_paths.append(final)

        # GET LABEL
        try:
            gold_list = ast.literal_eval(row['expected_order'])
            winner = gold_list[0]
            candidates = [row[f'image{k}_name'] for k in range(1, 6)]
            label = candidates.index(winner)
        except:
            label = 0

        # RETURN
        if self.mode == 'qwen':
            return {
                "image_paths": img_paths,
                "text": row['sentence'],
                "label": label
            }
        elif self.mode == 'fusion':
            tensors = []
            for p in img_paths:
                if p != "MISSING":
                    try:
                        img = Image.open(p).convert("RGB")
                        t = self.transform(img)
                    except:
                        t = torch.zeros(3, 224, 224)
                else:
                    t = torch.zeros(3, 224, 224)
                tensors.append(t)
            
            return {
                "pixel_values": torch.stack(tensors),
                "raw_text": row['sentence'],
                "labels": torch.tensor(label, dtype=torch.long)
            }
