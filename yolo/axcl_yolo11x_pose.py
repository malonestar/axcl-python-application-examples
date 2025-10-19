# axcl_yolo11x_pose.py
import cv2
import time
import argparse
import threading
import queue
from picamera2 import Picamera2
from axera_utils import PoseDetector

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
    parser.add_argument("-m", "--model", type=str, default="models/yolo11x-pose.axmodel", help="Path to YOLO Pose .axmodel.")
    parser.add_argument("--width", type=int, default=1280, help="Camera capture width.")
    parser.add_argument("--height", type=int, default=720, help="Camera capture height.")
    args = parser.parse_args()

    detector = PoseDetector(model_path=args.model)

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

    WINDOW_NAME = "LLM 8850 Pose Detection"
    cv2.namedWindow(WINDOW_NAME)
    
    print("[INFO] Starting pose detection feed. Press 'q' to quit.")
    prev_time = time.time()
    
    while True:
        try:
            original_frame, preprocessed_frame, ratio, pad_w, pad_h = input_queue.get(timeout=1)
        except queue.Empty:
            continue

        # Run pose inference
        poses = detector.infer_single(
            preprocessed_frame,
            original_frame.shape[:2],
            ratio,
            pad_w,
            pad_h
        )

        # Draw the poses (skeletons) on the frame
        result_frame = detector.draw_poses(original_frame, poses)

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(WINDOW_NAME, result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print("[INFO] Shutting down...")
    stop_event.set()
    preprocess_thread.join(timeout=1)
    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()