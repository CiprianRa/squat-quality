import pandas as pd
import numpy as np
from pathlib import Path

input_root = Path("../data_interpolated")
output_root = Path("../data_augmented")

def rotate_skeleton(coords, angle_deg):
    theta = np.deg2rad(angle_deg)
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ])
    coords = coords.reshape(-1, 2)
    rotated = coords @ rot_matrix.T
    return rotated.flatten()

def flip_skeleton(coords):
    coords = coords.reshape(-1, 2)
    coords[:, 0] *= -1  # Inversăm axa X
    return coords.flatten()

def add_noise(coords, std=0.01):
    noise = np.random.normal(0, std, coords.shape)
    return coords + noise

def augment_row(coords, aug_type):
    if aug_type == "flip":
        return flip_skeleton(coords)
    elif aug_type == "rot+10":
        return rotate_skeleton(coords, +10)
    elif aug_type == "rot-10":
        return rotate_skeleton(coords, -10)
    elif aug_type == "noise":
        return add_noise(coords)
    return coords

def process_file(csv_path, label_folder, filename_base, file_index):
    df = pd.read_csv(csv_path)
    if df.empty or len(df.columns) < 3:
        return

    # Separăm coloanele: frame, coordonate, features finale
    coord_cols = [col for col in df.columns if '_x' in col or '_y' in col]
    meta_cols = [col for col in df.columns if col not in coord_cols and col != "frame"]

    coords = df[coord_cols].values
    meta = df[meta_cols].values
    frame = df["frame"].values.reshape(-1, 1)

    augmentari = ["flip", "rot+10", "rot-10", "noise"]

    for i, aug in enumerate(augmentari, start=1):
        coords_aug = np.array([augment_row(row, aug) for row in coords])
        df_aug = pd.DataFrame(np.hstack([frame, coords_aug, meta]),
                              columns=["frame"] + coord_cols + meta_cols)

        out_dir = output_root / label_folder
        out_dir.mkdir(parents=True, exist_ok=True)

        output_name = f"{filename_base}_aug_{file_index}_{i}.csv"
        df_aug.to_csv(out_dir / output_name, index=False, float_format="%.4f")
        print(f"Salvat: {out_dir / output_name}")

def run_augmentation():
    for label_dir in input_root.iterdir():
        if not label_dir.is_dir():
            continue
        for csv_file in label_dir.glob("*.csv"):
            stem = csv_file.stem  # ex: corect_001
            parts = stem.split("_")
            if len(parts) < 2:
                print(f"Nume nevalid: {csv_file.name}")
                continue
            base_name = "_".join(parts[:-1])  # ex: corect
            index = parts[-1]                 # ex: 001

            process_file(
                csv_path=csv_file,
                label_folder=label_dir.name,
                filename_base=base_name,
                file_index=index
            )

if __name__ == "__main__":
    run_augmentation()
