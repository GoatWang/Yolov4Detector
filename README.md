# Introduction
Darknet python interface.

# Pre-Installation
1. darknet (please set the DARKNET_PATH env var)
2. following python oackages from pip
```
numpy == 1.17
opencv-python >= 4.1
matplotlib
pillow
```

# Installation
```
pip3 install -U --index-url http://192.168.0.128:28181/simple --trusted-host 192.168.0.128 Yolov4Detector

pip3 install -U --index-url http://rd.thinktronltd.com:28181/simple --trusted-host rd.thinktronltd.com Yolov4Detector
```

# Usage
## image
```
import cv2
from Yolov4Detector import io, Detector
from Yolov4Detector.utils import plot_one_box

# initialize Detector
cfg_fp, data_fp, weights_fp = io.get_params('yolov4') # yolov4_tiny
detector = Detector(cfg_fp, data_fp, weights_fp)
# img_fp = io.get_test_data('road1')
# img_fp = io.get_test_data('road2')
img_fp = io.get_test_data('road3')

image_bgr = cv2.imread(img_fp)
boxes, confs, clses = detector.detect(image_bgr, conf_thres=0.15, iou_thres=0.6)
if len(boxes) != 0:
    for xyxy, conf, cls in zip(boxes, confs, clses):
        image_bgr = plot_one_box(xyxy, image_bgr, label=cls, color=(255, 0, 0))
        print(xyxy, conf, cls)

cv2.imshow('img', image_bgr) 
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## video
```
import cv2
from datetime import datetime, timedelta
from Yolov4Detector import io, Detector
from Yolov4Detector.utils import plot_one_box

cfg_fp, data_fp, weights_fp = io.get_params('yolov4_tiny')
detector = Detector(cfg_fp, data_fp, weights_fp)
img_fp = io.get_test_data('road_video')

cap = cv2.VideoCapture(img_fp)
count = 0
st = datetime.now()
while(True):
    ret, image_bgr = cap.read()

    conf_thres = 0.15
    iou_thres = 0.6
    boxes, confs, clses = detector.detect(image_bgr, conf_thres=conf_thres, iou_thres=iou_thres)
    if boxes is not None:
        for xyxy, conf, cls in zip(boxes, confs, clses):
            plot_one_box(xyxy, image_bgr, label=cls, color=(255, 0, 0))


    cv2.imshow('frame', image_bgr)
    count += 1
    if datetime.now()- st > timedelta(seconds=1):
        print("fps:", count)
        count = 0
        st = datetime.now()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
```
