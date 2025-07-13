import motmetrics as mm
import pandas as pd

# Load GT and predictions
gt_file = "metrics/mot_metrics/tracking_eval/gt.csv"
pred_file = "metrics/mot_metrics/tracking_eval/pred.csv"

# Load CSVs
gt = pd.read_csv(gt_file, header=None, names=['FrameId', 'ObjectId', 'X', 'Y', 'Width', 'Height'])
pred = pd.read_csv(pred_file, header=None, names=['FrameId', 'ObjectId', 'X', 'Y', 'Width', 'Height'])

# Create an accumulator
acc = mm.MOTAccumulator(auto_id=True)

# Group by frames
gt_frames = gt.groupby('FrameId')
pred_frames = pred.groupby('FrameId')

all_frames = sorted(set(gt['FrameId']) | set(pred['FrameId']))

for frame in all_frames:
    gt_objects = gt_frames.get_group(frame) if frame in gt_frames.groups else pd.DataFrame(columns=gt.columns)
    pred_objects = pred_frames.get_group(frame) if frame in pred_frames.groups else pd.DataFrame(columns=pred.columns)

    gt_ids = gt_objects['ObjectId'].tolist()
    pred_ids = pred_objects['ObjectId'].tolist()

    gt_boxes = gt_objects[['X', 'Y']].values
    pred_boxes = pred_objects[['X', 'Y']].values

    distances = mm.distances.norm2squared_matrix(gt_boxes, pred_boxes, max_d2=0.04)

    acc.update(gt_ids, pred_ids, distances)

# Evaluate
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'idf1', 'num_switches', 'precision', 'recall'], name='summary')
print(summary)
