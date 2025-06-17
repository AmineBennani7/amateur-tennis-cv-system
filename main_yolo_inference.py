import os
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# Create the output folder if it doesn't exist
# This folder will be used to save the annotated images with detections
os.makedirs("outputs", exist_ok=True)

# ------------------- Normal Detection (no slicing) ------------------- #
def detect_normal(image_path, model_path="model/model1.pt", conf=0.1):
    # Load the custom YOLO model from the specified path
    model = YOLO(model_path)

    # Run prediction on the image
    results = model.predict(source=image_path, conf=conf, save=True)

    # Process the results
    for result in results:
        print("Detected classes:", result.names)
        print("Detected boxes:")
        for box in result.boxes:
            print(box)

        # Convert YOLO results to Supervision format
        detections = sv.Detections.from_ultralytics(result)

        # Read the original image using OpenCV
        image = cv2.imread(image_path)

        # Create labels with class name and confidence (e.g., "ball 0.85")
        labels = [
            f"{result.names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # Initialize the box and label annotators
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        # Annotate boxes and labels on the image
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        # Save the annotated image
        cv2.imwrite("outputs/normal_output.jpg", annotated_image)

# ------------------- Detection with Manual Slicing ------------------- #
def detect_with_slicing(image_path, model_path="model/model1.pt", slice_size=(384, 384), stride=(320, 320)):
    # Read the original image
    image = cv2.imread(image_path)

    # Load the YOLO model
    model = YOLO(model_path)

    # Get image dimensions
    height, width, _ = image.shape
    all_detections = []  # List to store all detections from slices

    # Apply sliding window across the entire image
    for y in range(0, height, stride[1]):
        for x in range(0, width, stride[0]):
            # Define the boundaries of the slice (sub-image)
            x_end = min(x + slice_size[0], width)
            y_end = min(y + slice_size[1], height)
            image_slice = image[y:y_end, x:x_end]  # Extract the slice

            # Run inference on the slice
            results = model.predict(image_slice, conf=0.1)
            result = results[0]

            # Convert results to Supervision format
            detections = sv.Detections.from_ultralytics(result)

            # Correct coordinates to map them to the full image
            detections.xyxy[:, [0, 2]] += x  # Adjust X-axis
            detections.xyxy[:, [1, 3]] += y  # Adjust Y-axis

            # Add detections to the list
            all_detections.append(detections)

    # Merge all detections from slices
    final_detections = sv.Detections.merge(all_detections)

    print("Total detections (with slicing):", len(final_detections.xyxy))

    # Create labels with class name and confidence
    labels = [
        f"{model.model.names[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(final_detections.class_id, final_detections.confidence)
    ]

    # Annotate the original image with all detections
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = box_annotator.annotate(scene=image, detections=final_detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=final_detections, labels=labels)

    # Save and display the annotated image
    cv2.imwrite("outputs/sliced_output.jpg", annotated_image)
    cv2.imshow("Detection with Slicing", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ------------------- EXECUTION ------------------- #
# Run both functions on the same image
# One version with normal detection, and another with slicing for improved detection of small objects

detect_normal("data/frames/video5/frame_0014.jpg")
detect_with_slicing("data/frames/video5/frame_0014.jpg")

