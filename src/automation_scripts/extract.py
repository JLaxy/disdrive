import cv2
import os

def extract_frames(video_path, output_folder, target_fps=5):
    os.makedirs(output_folder, exist_ok=True)
    
    video = cv2.VideoCapture(video_path)
    video_fps = video.get(cv2.CAP_PROP_FPS)

    if video_fps == 0:
        print(f"⚠️ Couldn't read FPS from {video_path}. Skipping...")
        return

    frame_interval = int(video_fps / target_fps)

    frame_count = 0
    saved_count = 0

    while True:
        success, frame = video.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    video.release()
    print(f"✅ Extracted {saved_count} frames from '{os.path.basename(video_path)}'")

def process_folder(input_folder, output_root, target_fps=5):
    supported_extensions = (".mp4", ".avi", ".mov", ".mkv")

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_extensions):
            video_path = os.path.join(input_folder, filename)
            video_name = os.path.splitext(filename)[0]
            output_folder = os.path.join(output_root, video_name)
            extract_frames(video_path, output_folder, target_fps)

# Example usage
input_folder = r"C:\Users\richa\Downloads\drive-download-20250425T142757Z-001" # Folder with input videos
output_root = r"C:\Users\richa\Desktop\extracted_frames"  # Where to store extracted frames
process_folder(input_folder, output_root, target_fps=5)
