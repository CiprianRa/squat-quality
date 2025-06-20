import os

folder = '../videos/Genunchi'
label = 'genunchi'

for i, filename in enumerate(sorted(os.listdir(folder)), 1):
    if filename.endswith('.mp4'):
        new_name = f"{label}_{i:03}.mp4"
        os.rename(os.path.join(folder, filename), os.path.join(folder, new_name))
