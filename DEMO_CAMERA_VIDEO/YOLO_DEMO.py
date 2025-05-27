import cv2
from ultralytics import YOLO

model = YOLO('best.pt')

choice = input("Enter 'c' for camera or 'v' for video file: ").strip().lower()

if choice == 'c':
    cap = cv2.VideoCapture(0)
elif choice == 'v':
    video_path = input("Enter the path to your video file: (For example «test.mp4») ")
    cap = cv2.VideoCapture(video_path)
else:
    print("Invalid choice. Exiting.")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    if choice == 'c':
        frame = cv2.flip(frame, 1) #Պատկերի հայելիականության վերացում(ուղղում)
    results = model.predict(frame, device='cpu')
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Inference        PRESS SPACE TO EXIT", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
