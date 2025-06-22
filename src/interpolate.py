import pandas as pd
import numpy as np
from pathlib import Path

input_root = Path("../data/raw/data_extracted")
output_root = Path("../data/data_interpolated")
target_frames = 60

def interpolate_csv_to_fixed_frames(df, target_frames=60):
    current_frames = df["frame"].values
    features = df.drop(columns=["frame"]).values

    # Dacă avem doar 1 frame → îl repetăm
    if len(current_frames) == 1:
        repeated = np.repeat(features, target_frames, axis=0)
        df_new = pd.DataFrame(repeated, columns=df.columns[1:])
        df_new.insert(0, "frame", np.arange(target_frames))
        return df_new

    # Dacă avem 0 frame-uri → ignoram
    if len(current_frames) < 1:
        return None

    # Interpolare normala
    interpolated = []
    new_frames = np.linspace(current_frames[0], current_frames[-1], num=target_frames)

    for i in range(features.shape[1]):
        interpolated.append(np.interp(new_frames, current_frames, features[:, i]))

    new_df = pd.DataFrame(np.stack(interpolated, axis=1), columns=df.columns[1:])
    new_df.insert(0, "frame", np.arange(target_frames))
    return new_df

def process_all_csvs():
    for label_dir in input_root.iterdir():
        if not label_dir.is_dir():
            continue

        for csv_file in label_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                df_interp = interpolate_csv_to_fixed_frames(df, target_frames)

                if df_interp is None:
                    print(f"Sărit (fără date): {csv_file}")
                    continue

                # Construieste aceeasi structura in output
                relative_folder = label_dir.name
                out_dir = output_root / relative_folder
                out_dir.mkdir(parents=True, exist_ok=True)

                output_path = out_dir / csv_file.name
                df_interp.to_csv(output_path, index=False, float_format="%.4f")
                print(f"Salvat: {output_path}")

            except Exception as e:
                print(f"Eroare la fisierul {csv_file}: {e}")

if __name__ == "__main__":
    process_all_csvs()
