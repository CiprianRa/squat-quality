import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

input_dirs = [Path("../data/data_interpolated"), Path("../data/data_augmented")]
output_dir = Path("../data")

split_ratios = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

def get_video_id(file_name: Path):
    """
    Extrage partea care identifica video-ul original, ex: calcaie_001
    """
    stem = file_name.stem
    parts = stem.split("_")
    return f"{parts[0]}_{parts[1]}"  # ex: calcaie_001

def collect_files_grouped_by_video():
    grouped = defaultdict(list)

    for input_root in input_dirs:
        for label_dir in input_root.iterdir():
            if not label_dir.is_dir():
                continue
            for file in label_dir.glob("*.csv"):
                video_id = get_video_id(file)
                rel_path_from_root = file.relative_to(input_root.parent)
                grouped[video_id].append(str(rel_path_from_root))
    return grouped

def split_and_save(grouped_files):
    all_video_ids = list(grouped_files.keys())
    random.shuffle(all_video_ids)

    total = len(all_video_ids)
    n_train = int(split_ratios["train"] * total)
    n_val = int(split_ratios["val"] * total)

    train_set = all_video_ids[:n_train]
    val_set = all_video_ids[n_train:n_train + n_val]
    test_set = all_video_ids[n_train + n_val:]

    splits = {
        "train_files.txt": train_set,
        "val_files.txt": val_set,
        "test_files.txt": test_set
    }

    for filename, video_id_list in splits.items():
        with open(output_dir / filename, "w") as f:
            for vid in video_id_list:
                for path in grouped_files[vid]:
                    f.write(f"{path}\n")
        print(f"Scris: {filename} ({len(video_id_list)} video-uri → {sum(len(grouped_files[v]) for v in video_id_list)} fisiere)")

if __name__ == "__main__":
    grouped = collect_files_grouped_by_video()
    split_and_save(grouped)
