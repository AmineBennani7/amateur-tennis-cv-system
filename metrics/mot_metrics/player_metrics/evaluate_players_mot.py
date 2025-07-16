import os
import cv2
import motmetrics as mm
import pandas as pd

# File paths
gt_file = "metrics/mot_metrics/player_metrics/tracking_eval/gt.csv"
pred_file = "metrics/mot_metrics/player_metrics/tracking_eval/pred.csv"
IMG_FOLDER = "metrics/tennis_players_eval_dataset/images"
OUTPUT_VIS_FOLDER = "metrics/mot_metrics/player_metrics/visual_tracking_frames"
os.makedirs(OUTPUT_VIS_FOLDER, exist_ok=True)

# Load ground truth and prediction data
gt = pd.read_csv(gt_file, header=None, names=['FrameId', 'ObjectId', 'X', 'Y', 'Width', 'Height'])
pred = pd.read_csv(pred_file, header=None, names=['FrameId', 'ObjectId', 'X', 'Y', 'Width', 'Height'])

# Initialize MOT accumulator
acc = mm.MOTAccumulator(auto_id=True)
gt_frames = gt.groupby('FrameId')
pred_frames = pred.groupby('FrameId')
all_frames = sorted(set(gt['FrameId']) | set(pred['FrameId']))

# Process frame-by-frame
for frame in all_frames:
    gt_objs = gt_frames.get_group(frame) if frame in gt_frames.groups else pd.DataFrame(columns=gt.columns)
    pred_objs = pred_frames.get_group(frame) if frame in pred_frames.groups else pd.DataFrame(columns=pred.columns)

    gt_ids = list(range(len(gt_objs)))
    pred_ids = list(range(len(pred_objs)))

    gt_centers = gt_objs[['X', 'Y']].values
    pred_centers = pred_objs[['X', 'Y']].values

    # Compute distance matrix and update accumulator
    distances = mm.distances.norm2squared_matrix(gt_centers, pred_centers, max_d2=0.04)
    acc.update(gt_ids, pred_ids, distances)

# Compute tracking metrics
mh = mm.metrics.create()
summary = mh.compute(
    acc,
    metrics=['num_frames', 'mota', 'motp', 'precision', 'recall', 'num_switches'],
    name='player_tracking'
)
print(summary)

# Visualization 

print(" Generating tracking visualizations...")

for frame in all_frames:
    frame_str = str(frame)
    image_name = None

    for fname in os.listdir(IMG_FOLDER):
        if frame_str in fname:
            image_name = fname
            break

    if image_name is None:
        print(f"No image found for frame: {frame}")
        continue

    img_path = os.path.join(IMG_FOLDER, image_name)
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]

    # Draw ground truth boxes in green
    if frame in gt_frames.groups:
        for _, row in gt_frames.get_group(frame).iterrows():
            x1 = int((row.X - row.Width / 2) * w)
            y1 = int((row.Y - row.Height / 2) * h)
            x2 = int((row.X + row.Width / 2) * w)
            y2 = int((row.Y + row.Height / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"GT-{int(row.ObjectId)}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # Draw predicted boxes in red
    if frame in pred_frames.groups:
        for _, row in pred_frames.get_group(frame).iterrows():
            x1 = int((row.X - row.Width / 2) * w)
            y1 = int((row.Y - row.Height / 2) * h)
            x2 = int((row.X + row.Width / 2) * w)
            y2 = int((row.Y + row.Height / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"Pred-{int(row.ObjectId)}", (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    out_path = os.path.join(OUTPUT_VIS_FOLDER, f"{frame}_{image_name}")
    cv2.imwrite(out_path, img)

print(f"Tracking visualizations saved in: {OUTPUT_VIS_FOLDER}")
