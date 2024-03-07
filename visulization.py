import cv2
from pathlib import Path
import glob
import json
import time

frame_dir = Path("/home/liuyang/datasets/qiyuan/240208/20240208-01/selected/occlusion/")
label_f = "./outputs/bbox.txt"


mode = "rgb"


img_files = glob.glob(str(frame_dir / "*.jpg"))
img_files.sort()

start = False
start_frame_id = 0
rec_res = []
with open(label_f, 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        rec = line.strip().split()
        rec = [int(_) for _ in rec]
        rec_res.append(rec)

for frame_id in range(len(img_files)):
    frame = cv2.imread(img_files[frame_id])
    print(frame_id)
    if frame_id >= start_frame_id:
        rec = rec_res[frame_id - start_frame_id]
        cv2.rectangle(frame, (int(rec[0]), int(rec[1])), (int(rec[0] + rec[2]), int(rec[1]) + int(rec[3])), (0, 255, 255))
    frame = frame[1000:1400, 1140:1940]
    cv2.imshow("video_name", frame)

    key = cv2.waitKey(1)
    while not start:
        key = cv2.waitKey(1)
        if key == ord("s"):
            start = True
            break
        time.sleep(0.5)

    if key == ord("q"):
        break

cv2.destroyAllWindows()
