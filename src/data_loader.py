import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

# load video frames from a folder, selecting a fixed number of frames per video
def load_video_frames(video_folder, num_frames=16, resize=(224, 224)):
    frame_files = sorted([f for f in os.listdir(video_folder) if f.endswith(".jpg")])
    # evenly sample `num_frames` indices from the full list
    selected = np.linspace(0, len(frame_files) - 1, num_frames).astype(int)
    
    frames = []
    for idx in selected:
        frame_path = os.path.join(video_folder, frame_files[idx])
        img = cv2.imread(frame_path)
        img = cv2.resize(img, resize) # resize to standard input size
        img = img[..., ::-1] / 255.0 # convert BGR to RGB and normalize
        frames.append(img)
    
    return np.stack(frames, axis=0) # shape is (T, H, W, C)

# load a CSV containing (video_id, caption) pairs
def load_caption_csv(path):
    return pd.read_csv(path)

# pair each video's frames with its corresponding caption
def load_dataset(video_root, caption_csv, num_frames=16):
    caption_df = load_caption_csv(caption_csv)
    dataset = []

    # iterate through all caption entries
    for _, row in tqdm(caption_df.iterrows(), total=len(caption_df)):
        video_id = row['video_id']
        caption = row['caption']
        video_folder = os.path.join(video_root, video_id)

        # only process if the video folder exists
        if os.path.exists(video_folder):
            try:
                frames = load_video_frames(video_folder, num_frames)
                dataset.append((frames, caption))
            except Exception as e:
                print(f"Error processing {video_id}: {e}")
    
    return dataset # list of (video_tensor, caption) pairs