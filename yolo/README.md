# YOLO Object Detection Applications on M5Stack LLM 8850 Accelerator  

## axcl_yolo11x.py  
This script runs live inferece on a camera using yolo11x.axmodel

```
python axcl_yolo11x.py
```
Opens an OpenCV window with yolo11x inference on the camera feed.  
<img src="https://i.imgur.com/ZT1k3uD.png" alt="YOLO object detection demo" width=50%>  

## axcl_yolo11x_pose.py  

Same OpenCV and camera as the previous example, but this runs yolo11x-pose for pose detection.  
```
python axcl_yolo11x_pose.py
```
<img src="https://i.imgur.com/E4egCy6.png" alt="YOLO pose detection demo" width=50%>  

## axcl_yolo11x_trigger.py  

```
python axcl_yolo11x_trigger.py  
```

This one is fun.  Run the script, then the instructions will appear on the screen.  Press 'd' to draw an ROI on the screen.  Press 's' to save.  
When a person is detected in the ROI for 5.0 seconds, a screenshot is saved and sent to your personal discord server via an api hook.  

You will need to open notify.py and add your discord hook in place of 'YOUR-DISCORD-HOOK-HERE'.  
The notify.py script contains the functions for sending messages and images to discord via api hook.  

<img src="https://i.imgur.com/FL21mEm.png" alt="YOLO detection trigger demo" width=75%>  

<img src="https://i.imgur.com/dLgjvsu.png" alt="YOLO detection trigger demo" width=75%>  

<img src="https://i.imgur.com/5hkDqRS.png" alt="YOLO detection trigger demo" width=75%>  

<img src="https://i.imgur.com/Tj5dgNQ.jpeg" alt="YOLO detection trigger demo" width=75%>  











