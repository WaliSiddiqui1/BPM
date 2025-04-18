import os
import cv2
import json
from pathlib import Path
from yt_dlp import YoutubeDL
import webvtt

# Set up directories
VIDEO_DIR = "videos"
CLIP_DIR = "data/clips"
CAPTION_DIR = "data/captions"
FRAME_DIR = "data/frames"
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(CLIP_DIR, exist_ok=True)
os.makedirs(CAPTION_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)

# Download YouTube videos and captions
def download_video_and_captions(youtube_url):
    ydl_opts = {
        'format': 'bestvideo+bestaudio',
        'outtmpl': os.path.join(VIDEO_DIR, '%(title)s.%(ext)s'),
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'skip_download': False
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

# Segment video into short clips
def segment_video(video_path, clip_duration=5):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    length_in_sec = total_frames // fps
    video_name = Path(video_path).stem

    clip_paths = []
    for i in range(0, length_in_sec, clip_duration):
        cap.set(cv2.CAP_PROP_POS_MSEC, i * 1000)
        out_path = os.path.join(CLIP_DIR, f"{video_name}_clip_{i}.mp4")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(3)), int(cap.get(4))))
        for _ in range(clip_duration * fps):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()
        clip_paths.append((out_path, i))
    cap.release()
    return clip_paths

# Extract frames from a video
def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = [int(x) for x in list(
        map(lambda x: x * total_frames // num_frames, range(num_frames))
    )]
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    cap.release()
    return frames

# Parse captions
def parse_captions(vtt_path):
    captions = []
    for caption in webvtt.read(vtt_path):
        if caption.text.strip():
            start = caption.start
            end = caption.end
            text = caption.text.strip().replace('\n', ' ')
            captions.append((start, end, text))
    return captions

# Align captions with clips
def timestamp_to_seconds(ts):
    h, m, s = ts.split(':')
    s, ms = s.split('.')
    return int(h)*3600 + int(m)*60 + int(s)

def align_captions_to_clips(captions, clip_start, clip_end):
    result = []
    for start, end, text in captions:
        start_s = timestamp_to_seconds(start)
        end_s = timestamp_to_seconds(end)
        if clip_start <= start_s < clip_end:
            result.append(text)
    return " ".join(result)

# Full pipeline
def process_video(video_path, vtt_path):
    video_name = Path(video_path).stem
    captions = parse_captions(vtt_path)
    clips = segment_video(video_path)

    for clip_path, start_time in clips:
        end_time = start_time + 5
        aligned_caption = align_captions_to_clips(captions, start_time, end_time)

        if not aligned_caption.strip():
            continue

        frames = extract_frames(clip_path)
        frames_out_path = os.path.join(FRAME_DIR, f"{Path(clip_path).stem}.json")
        frame_data = [frame.tolist() for frame in frames]

        data_entry = {
            "clip_id": Path(clip_path).stem,
            "frames": frame_data,
            "caption": aligned_caption,
            "metadata": {
                "source_video": video_name,
                "start_time": start_time,
                "end_time": end_time
            }
        }

        with open(frames_out_path, 'w') as f:
            json.dump(data_entry, f)

if __name__ == "__main__":
    # Replace this with your actual YouTube video URL
    example_youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    download_video_and_captions(example_youtube_url)

    # Process all downloaded videos
    for file in os.listdir(VIDEO_DIR):
        if file.endswith(".mp4"):
            base_name = Path(file).stem
            vtt_file = os.path.join(VIDEO_DIR, base_name + ".en.vtt")
            video_file = os.path.join(VIDEO_DIR, file)
            if os.path.exists(vtt_file):
                process_video(video_file, vtt_file)