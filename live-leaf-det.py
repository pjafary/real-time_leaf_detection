import cv2
from ultralyticsplus import YOLO, render_result
from PIL import Image
import numpy as np

# Load the YOLO model
model = YOLO('foduucom/plant-leaf-detection-and-classification')

# Set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # Maximum number of detections per image

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is typically the default camera

# Check if webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop to read frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform inference on the frame
    results = model.predict(frame)

    # Render results on the frame
    render = render_result(model=model, image=frame, result=results[0])
    
    # Check if render_result output is a PIL Image, then convert to OpenCV format
    if isinstance(render, Image.Image):
        #render = cv2.cvtColor(np.array(render), cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV display
        render = np.array(render)


    # Display the resulting frame
    cv2.imshow('YOLO Real-Time Detection', render)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
