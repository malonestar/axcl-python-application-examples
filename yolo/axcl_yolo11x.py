# axcl_yolo11x.py

import cv2
import time
import argparse
import threading
import queue
from picamera2 import Picamera2
from axera_utils import Detector

def preprocess_worker(picam2, detector, input_queue, stop_event):
    print("[INFO] Preprocess worker started.")
    while not stop_event.is_set():
        frame_rgb = picam2.capture_array("lores")
        input_img, ratio, pad_w, pad_h = detector._letterbox(frame_rgb)
        try:
            input_queue.put((frame_rgb, input_img, ratio, pad_w, pad_h), block=False)
        except queue.Full:
            continue
    print("[INFO] Preprocess worker stopped.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="models/yolo11x.axmodel", help="Path to .axmodel.")
    parser.add_argument("-l", "--labels", type=str, default="coco.txt", help="Path to labels file.")
    parser.add_argument("--width", type=int, default=1280, help="Camera capture width.")
    parser.add_argument("--height", type=int, default=720, help="Camera capture height.")
    args = parser.parse_args()

    detector = Detector(model_path=args.model, labels_path=args.labels)

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (2304, 1296)},
        lores={"size": (args.width, args.height), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    input_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    preprocess_thread = threading.Thread(
        target=preprocess_worker,
        args=(picam2, detector, input_queue, stop_event),
        daemon=True
    )
    preprocess_thread.start()

    print("[INFO] Starting pipelined camera feed. Press 'q' to exit.")
    
    prev_time = time.time()
    
    while True:
        try:
            original_frame, preprocessed_frame, ratio, pad_w, pad_h = input_queue.get(timeout=1)
        except queue.Empty:
            print("[WARN] Input queue is empty. Is the camera running?")
            continue

        detections = detector.infer_single(
            preprocessed_frame,
            original_frame.shape[:2],
            ratio,
            pad_w,
            pad_h
        )

        result_frame = detector.draw_detections(original_frame, detections)

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("LLM 8850 Object Detection", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print("[INFO] Shutting down...")
    stop_event.set()
    preprocess_thread.join(timeout=1)
    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()