from ultralytics import YOLO
import cv2

def main():
    model = YOLO("runs/detect/train/weights/best.pt")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Kamera tidak dapat dibuka")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        annotated_frame = results[0].plot()

        cv2.imshow("Computer Vision - YOLO + Roboflow", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
