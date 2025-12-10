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
        """
        Args:
            data_root_path: Path where you unzipped the data (e.g. /content/admire_data)
            split: 'Train', 'Dev', or 'Test'
            mode: 'qwen' (returns file paths) or 'fusion' (returns pixel tensors)
        """
        self.mode = mode
        
        # 1. AUTO-FIND TSV
        print(f"🕵️ Scanning {data_root_path} for {split} data...")
        tsv_candidates = glob.glob(os.path.join(data_root_path, "**", f"*{split}*.tsv"), recursive=True)
        
        if not tsv_candidates:
             # Fallback: Find ANY tsv if specific split not found
             tsv_candidates = glob.glob(os.path.join(data_root_path, "**", "*.tsv"), recursive=True)
        
        if not tsv_candidates:
            raise FileNotFoundError(f"❌ TSV file not found in {data_root_path}")
            
        self.tsv_path = tsv_candidates[0]
        self.data_root = os.path.dirname(self.tsv_path) # Images are anchored to the TSV location
        
        print(f"✅ Loaded: {self.tsv_path}")
        self.df = pd.read_csv(self.tsv_path, sep='\t')
        
        # 2. TRANSFORM (Only used if mode='fusion')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        idiom_folder = str(row['compound'])
        
        # 3. ROBUST IMAGE FINDING
        img_paths = []
        for i in range(1, 6):
            fname = row[f'image{i}_name']
            
            # Try Path A: root/idiom/filename (Standard)
            path_a = os.path.join(self.data_root, idiom_folder, fname)
            # Try Path B: root/filename (Fallback if flattened)
            path_b = os.path.join(self.data_root, fname)
            
            if os.path.exists(path_a): final = path_a
            elif os.path.exists(path_b): final = path_b
            else: final = "MISSING"
            
            img_paths.append(final)

        # 4. GET LABEL
        try:
            gold_list = ast.literal_eval(row['expected_order'])
            winner = gold_list[0]
            candidates = [row[f'image{k}_name'] for k in range(1, 6)]
            label = candidates.index(winner)
        except:
            label = 0

        # 5. RETURN BASED ON MODE
        # --- MODE: QWEN ---
        if self.mode == 'qwen':
            return {
                "image_paths": img_paths,
                "text": row['sentence'],
                "label": label
            }
            
        # --- MODE: FUSION ---
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
            
            # Stack 5 images -> [5, 3, 224, 224]
            return {
                "pixel_values": torch.stack(tensors),
                "raw_text": row['sentence'],
                "labels": torch.tensor(label, dtype=torch.long)
            }