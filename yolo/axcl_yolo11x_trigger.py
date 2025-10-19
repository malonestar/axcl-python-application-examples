# axcl_yolo11x_trigger.py
import cv2
import time
import argparse
import threading
import queue
import os
from pathlib import Path
from picamera2 import Picamera2
from axera_utils import Detector
from notify import send_discord_message, send_discord_image

detect_mode = False
roi_defined = False
roi_box = None
drawing = False
start_point = None

def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, roi_box, detect_mode
    if not detect_mode: return
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True; start_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        param['temp_box'] = (start_point[0], start_point[1], x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False; end_point = (x, y)
        roi_box = (min(start_point[0], end_point[0]), min(start_point[1], end_point[1]), max(start_point[0], end_point[0]), max(start_point[1], end_point[1]))
        param['temp_box'] = None
        print(f"[INFO] ROI box drawn at {roi_box}. Press 's' to save or 'c' to cancel.")

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
    global detect_mode, roi_defined, roi_box, drawing, start_point

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="models/yolo11x.axmodel", help="Path to .axmodel.")
    parser.add_argument("-l", "--labels", type=str, default="coco.txt", help="Path to labels file.")
    parser.add_argument("--width", type=int, default=1280, help="Camera capture width.")
    parser.add_argument("--height", type=int, default=720, help="Camera capture height.")
    args = parser.parse_args()
    detector = Detector(model_path=args.model, labels_path=args.labels)
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (2304, 1296)}, lores={"size": (args.width, args.height), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    input_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    preprocess_thread = threading.Thread(target=preprocess_worker, args=(picam2, detector, input_queue, stop_event), daemon=True)
    preprocess_thread.start()
    
    WINDOW_NAME = "LLM 8850 Object Detection"
    callback_param = {'temp_box': None}
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, callback_param)

    presence_start_time = None
    presence_triggered_this_event = False
    MIN_PRESENCE_SECONDS = 5
    TARGET_LABEL = "person"
    output_path = Path('output')
    output_path.mkdir(exist_ok=True)

    print("[INFO] Starting feed. Press 'd' to draw ROI, 'q' to quit.")
    prev_time = time.time()
    
    while True:
        try:
            original_frame, preprocessed_frame, ratio, pad_w, pad_h = input_queue.get(timeout=1)
        except queue.Empty:
            continue

        detections = detector.infer_single(preprocessed_frame, original_frame.shape[:2], ratio, pad_w, pad_h)
        result_frame = detector.draw_detections(original_frame.copy(), detections)
        
        person_in_roi = False
        if roi_defined and roi_box:
            for det in detections:
                if detector.labels[det.label] == TARGET_LABEL and det.prob > 0.5:
                    x, y, w, h = det.bbox
                    det_center_x, det_center_y = x + w / 2, y + h / 2
                    if roi_box[0] < det_center_x < roi_box[2] and roi_box[1] < det_center_y < roi_box[3]:
                        person_in_roi = True
                        break

        current_time = time.time()
        if person_in_roi:
            if presence_start_time is None:
                presence_start_time = current_time
            elif (current_time - presence_start_time >= MIN_PRESENCE_SECONDS) and not presence_triggered_this_event:
                presence_triggered_this_event = True
                print(f"[ALERT] Person detected in ROI for {MIN_PRESENCE_SECONDS} seconds!")
                
                timestamp = int(time.time())
                screenshot_path = output_path / f"alert_{timestamp}.jpg"
                
                image_to_save_with_boxes = detector.draw_detections(original_frame.copy(), detections)
                cv2.imwrite(str(screenshot_path), image_to_save_with_boxes)
                
                send_discord_message(f"Person detected in Region of Interest!")
                send_discord_image(str(screenshot_path), message="Detection snapshot attached.")
        else:
            presence_start_time = None
            presence_triggered_this_event = False

        if roi_defined and roi_box:
            cv2.rectangle(result_frame, (roi_box[0], roi_box[1]), (roi_box[2], roi_box[3]), (0, 0, 255), 2)
        elif 'temp_box' in callback_param and callback_param['temp_box']:
             temp = callback_param['temp_box']
             cv2.rectangle(result_frame, (temp[0], temp[1]), (temp[2], temp[3]), (0, 255, 255), 2)
        if detect_mode:
            cv2.putText(result_frame, "Draw ROI. 's' to save, 'c' to cancel.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(result_frame, "Press 'd' to define ROI. 'q' to quit.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if presence_start_time and not presence_triggered_this_event:
            seconds_in_roi = current_time - presence_start_time
            cv2.putText(result_frame, f"Person in ROI: {seconds_in_roi:.1f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(WINDOW_NAME, result_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('d'):
            print("[INFO] Detect mode enabled. Draw ROI.")
            detect_mode, roi_defined, roi_box = True, False, None
        elif key == ord('s') and detect_mode:
            if roi_box:
                detect_mode, roi_defined = False, True
                print(f"[INFO] ROI saved: {roi_box}")
            else:
                detect_mode = False
                print("[WARN] No ROI drawn. Exiting detect mode.")
        elif key == ord('c') and detect_mode:
            detect_mode, roi_box, callback_param['temp_box'] = False, None, None
            print("[INFO] ROI cancelled.")
    
    print("[INFO] Shutting down...")
    stop_event.set()
    preprocess_thread.join(timeout=1)
    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()