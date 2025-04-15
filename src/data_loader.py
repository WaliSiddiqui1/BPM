import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

def load_video_frames(video_folder, num_frames=16, resize=(224, 224)):
    frame_files = sorted([f for f in os.listdir(video_folder) if f.endswith(".jpg")])
    selected = np.linspace(0, len(frame_files) - 1, num_frames).astype(int)
    
    frames = []
    for idx in selected:
        frame_path = os.path.join(video_folder, frame_files[idx])
        img = cv2.imread(frame_path)
        img = cv2.resize(img, resize)
        img = img[..., ::-1] / 255.0
        frames.append(img)
    
    return np.stack(frames, axis=0)

def load_caption_csv(path):
    return pd.read_csv(path)

def load_dataset(video_root, caption_csv, num_frames=16):
    caption_df = load_caption_csv(caption_csv)
    dataset = []

    for _, row in tqdm(caption_df.iterrows(), total=len(caption_df)):
        video_id = row['video_id']
        caption = row['caption']
        video_folder = os.path.join(video_root, video_id)
        if os.path.exists(video_folder):
            try:
                frames = load_video_frames(video_folder, num_frames)
                dataset.append((frames, caption))
            except Exception as e:
                print(f"Error processing {video_id}: {e}")
    
    return dataset