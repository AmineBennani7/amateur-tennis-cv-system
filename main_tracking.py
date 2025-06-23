import cv2
from trackers.player_tracker import PlayerTracker

# Initialize the player tracker with your model
player_tracker = PlayerTracker("model/model1.pt")  # ← path to your YOLOv8 model

# Load the video
cap = cv2.VideoCapture("data/cropped_videos/video1_cropped.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Detect players in the current frame
    detections = player_tracker.detect_frame(frame)

    # Debug print to check what detections are returned
    print("Detections:", detections)

    # Draw bounding boxes only if there are any detections
    if detections:
        frame = player_tracker.draw_bboxes([frame], [detections])[0]

    # Display the result
    print("Frame processed. Displaying result.")

    cv2.imshow("Player Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break  # Press 'q' to exit

# Release video capture and close display window
cap.release()
cv2.destroyAllWindows()
