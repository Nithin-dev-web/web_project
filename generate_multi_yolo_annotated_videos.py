import cv2
import os
import glob

# Base directory where videos and folders are located
BASE_DIR = r"C:\Users\Nithin Kumar G\Downloads\Vendor_video"

# Get all MP4 videos in base directory
videos = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".mp4")]
print(f"Found {len(videos)} video(s) to process.\n")

for video_name in videos:
    print(f"🎬 Processing: {video_name}")
    video_path = os.path.join(BASE_DIR, video_name)

    folder_name = os.path.splitext(video_name)[0]
    video_folder = os.path.join(BASE_DIR, folder_name)
    frame_folder = os.path.join(video_folder, "obj_train_data")

    # Load class names
    names_path = os.path.join(video_folder, "obj.names")
    CLASS_NAMES = []
    if os.path.exists(names_path):
        with open(names_path, "r") as f:
            CLASS_NAMES = [line.strip() for line in f if line.strip()]
        print(f"✅ Loaded {len(CLASS_NAMES)} class names.")
    else:
        print(f"⚠️ 'obj.names' not found — labels will be numeric only.")

    # YOLO annotation files
    txt_files = sorted(glob.glob(os.path.join(frame_folder, "*.txt")))
    if not txt_files:
        print(f"⚠️ No annotation files in {frame_folder}, skipping...\n")
        continue

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open {video_name}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"🎞️ Video resolution: {vid_width}x{vid_height}, {fps:.2f} FPS")

    output_path = os.path.join(BASE_DIR, f"{folder_name}_annotated_fixed.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid_width, vid_height))

    # Color map for specific classes
    COLOR_MAP = {
        "person": (0, 255, 0),      # Green
        "long gun": (0, 0, 255),    # Red
        "short gun": (255, 0, 0)    # Blue
    }

    frame_id = 0
    for txt_file in txt_files:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id, x, y, w, h = map(float, parts[:5])

                # Convert YOLO normalized coordinates to pixel coordinates
                x1 = int((x - w / 2) * vid_width)
                y1 = int((y - h / 2) * vid_height)
                x2 = int((x + w / 2) * vid_width)
                y2 = int((y + h / 2) * vid_height)

                label = CLASS_NAMES[int(cls_id)] if CLASS_NAMES and int(cls_id) < len(CLASS_NAMES) else str(int(cls_id))
                color = COLOR_MAP.get(label.lower(), (0, 255, 255))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"Frame: {frame_id}/{total_frames}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Annotated video saved: {output_path}\n")

print("🎉 All videos processed successfully!")
