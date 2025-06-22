import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing
import gc

mp_pose = mp.solutions.pose

input_root = Path("../data/raw/videos")
output_root = Path("../data/raw/data_extracted")

KEYPOINT_NAMES = {
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
    "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29, "RIGHT_HEEL": 30,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_FOOT": 31, "RIGHT_FOOT": 32
}


def get_point(landmarks, name):
    idx = KEYPOINT_NAMES[name]
    p = landmarks[idx]
    return np.array([p.x, p.y]), p.visibility


def normalize_landmarks(landmarks, ref_mid_hip, ref_femur_len):
    try:
        points = {}
        for name in KEYPOINT_NAMES:
            coord, _ = get_point(landmarks, name)
            points[name] = (coord - ref_mid_hip) / ref_femur_len

        hip_heel_dist = np.linalg.norm(points["LEFT_HIP"] - points["LEFT_HEEL"])
        hip_ankle_dist = np.linalg.norm(points["LEFT_HIP"] - points["LEFT_ANKLE"])
        shoulder_heel_dist = np.linalg.norm(points["LEFT_SHOULDER"] - points["LEFT_HEEL"])
        shoulder_gap = abs(points["LEFT_SHOULDER"][0] - points["RIGHT_SHOULDER"][0])

        features = []
        for name in KEYPOINT_NAMES:
            features.extend(points[name])
        features.extend([hip_heel_dist, hip_ankle_dist, shoulder_heel_dist, shoulder_gap])
        return features
    except:
        return None


def process_video(video_info):
    video_path, label = video_info
    output_label_dir = output_root / label
    output_label_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_id = 0
    data = []
    ref_mid_hip = None
    ref_femur_len = None

    with mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                if ref_mid_hip is None:
                    l_hip, l_vis = get_point(landmarks, "LEFT_HIP")
                    r_hip, r_vis = get_point(landmarks, "RIGHT_HIP")
                    if l_vis >= 0.5 and r_vis >= 0.5:
                        ref_mid_hip = (l_hip + r_hip) / 2
                    elif l_vis >= 0.5:
                        ref_mid_hip = l_hip
                    elif r_vis >= 0.5:
                        ref_mid_hip = r_hip

                if ref_femur_len is None and ref_mid_hip is not None:
                    l_knee, l_knee_vis = get_point(landmarks, "LEFT_KNEE")
                    r_knee, r_knee_vis = get_point(landmarks, "RIGHT_KNEE")
                    if l_vis >= 0.5 and l_knee_vis >= 0.5:
                        ref_femur_len = np.linalg.norm(l_hip - l_knee)
                    elif r_vis >= 0.5 and r_knee_vis >= 0.5:
                        ref_femur_len = np.linalg.norm(r_hip - r_knee)

                if ref_mid_hip is not None and ref_femur_len is not None and ref_femur_len >= 1e-5:
                    row = normalize_landmarks(landmarks, ref_mid_hip, ref_femur_len)
                    if row:
                        if not data:
                            print(f"{video_path.name}: primul frame valid este {frame_id}")
                        data.append([frame_id] + row)

            frame_id += 1

    cap.release()
    del cap
    gc.collect()

    if data:
        columns = ["frame"]
        for name in KEYPOINT_NAMES:
            columns.extend([f"{name.lower()}_x", f"{name.lower()}_y"])
        columns += ["hip_heel_dist", "hip_ankle_dist", "shoulder_heel_dist", "shoulder_gap"]

        df = pd.DataFrame(data, columns=columns)
        output_file = output_label_dir / (video_path.stem + ".csv")
        df.to_csv(output_file, index=False, float_format="%.4f")
        print(f"Salvat: {output_file}")
    else:
        print(f"Niciun frame valid: {video_path}")


if __name__ == "__main__":
    all_video_paths = []
    for label_dir in input_root.iterdir():
        if not label_dir.is_dir():
            continue
        for video_file in label_dir.glob("*.mp4"):
            all_video_paths.append((video_file, label_dir.name))

    with multiprocessing.Pool(processes=8) as pool:
        pool.map(process_video, all_video_paths)
