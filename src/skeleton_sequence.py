import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import numpy as np

class SkeletonSequenceDataset(Dataset):
    def __init__(self, root_dir, file_list_path, label_map=None):
        self.root_dir = Path(root_dir)
        with open(file_list_path, "r") as f:
            self.file_paths = [line.strip() for line in f if line.strip()]

        # Eticheta este folderul imediat urmator dupa root (ex: calcaie)
        self.labels = [Path(path).parts[1] for path in self.file_paths]
        self.label_to_idx = label_map or {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

        print("Etichete detectate:", self.label_to_idx)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        rel_path = self.file_paths[idx]
        full_path = self.root_dir / rel_path

        df = pd.read_csv(full_path)
        data = df.drop(columns=["frame"]).values.astype(np.float32)

        label_name = Path(rel_path).parts[1]  # extrage: calcaie, genunchi etc.
        label_idx = self.label_to_idx[label_name]

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label_idx, dtype=torch.long)
