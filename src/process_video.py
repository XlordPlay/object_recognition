# src/process_video.py
import cv2
from detect_bags import BagDetector

def main(input_video, output_video):
    # Инициализация детектора
    detector = BagDetector('models/yolov3.cfg', 'models/yolov3.weights', 'data/obj.names')

    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        for (box, class_id, confidence) in detections:
            x, y, w, h = box
            label = f"{detector.classes[class_id]}: {confidence:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main('data/input_video/video.mp4', 'data/output_video/processed_video.mp4')