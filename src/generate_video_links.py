from yt_dlp import YoutubeDL

playlist_url = "https://www.youtube.com/playlist?list=PL5j8RirTTnK5rfAPFJFwaqJvLweQynhjq"
output_file = "video_links.txt"

ydl_opts = {
    'quiet': True,
    'extract_flat': True,
    'skip_download': True
}

with YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(playlist_url, download=False)
    video_urls = [f"https://www.youtube.com/watch?v={entry['id']}" for entry in info['entries']]

with open(output_file, "w") as f:
    for url in video_urls:
        f.write(url + "\n")

print(f"Saved {len(video_urls)} video links to {output_file}")