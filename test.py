import os
import cv2
import unittest
from datetime import datetime, timedelta
from Yolov4Detector import io, Detector
from Yolov4Detector.utils import plot_one_box

class TestYolov4Detector(unittest.TestCase):
    # def setUp(self):
    #     time.sleep(1)

    # def tearDown(self):
    #     shutil.rmtree(self.output_dir)
    #     time.sleep(1)

    #def test_detector_yolov4(self):
    #    cfg_fp, data_fp, weights_fp = io.get_params('yolov4')
    #    detector = Detector(cfg_fp, data_fp, weights_fp)
    #    img_fp = io.get_test_data('road3')

    #    image_bgr = cv2.imread(img_fp)
    #    boxes, confs, clses = detector.detect(image_bgr, conf_thres=0.15, iou_thres=0.6)
    #    if len(boxes) != 0:
    #        for xyxy, conf, cls in zip(boxes, confs, clses):
    #            plot_one_box(xyxy, image_bgr, label=cls, color=(255, 0, 0))
    #            print(xyxy, conf, cls)

    #    cv2.imshow('img', image_bgr) 
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()

    def test_detector_yolov4_tiny(self):
        cfg_fp, names_fp, weights_fp = io.get_test_params()
        detector = Detector(cfg_fp, names_fp, weights_fp)
        img_fp = io.get_test_data('bus')
        img_fp = io.get_test_data('zidane')

        image_bgr = cv2.imread(img_fp)
        boxes, confs, clses = detector.detect(image_bgr, conf_thres=0.15, iou_thres=0.6)
        if len(boxes) != 0:
            for xyxy, conf, cls in zip(boxes, confs, clses):
                plot_one_box(xyxy, image_bgr, label=cls, color=(255, 0, 0))
                print(xyxy, conf, cls)

        cv2.imshow('img', image_bgr) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # def test_detector_batch_yolov4(self):
    #     cfg_fp, data_fp, weights_fp = io.get_params('yolov4')
    #     detector = Detector(cfg_fp, data_fp, weights_fp)
    #     img_fps = [io.get_test_data('road1'), io.get_test_data('road2'), io.get_test_data('road3')]
    #     images_bgr = [cv2.imread(img_fp) for img_fp in img_fps]

    #     preds = detector.detect_batch(images_bgr, conf_thres=0.1, iou_thres=0.1, batch_size=2)
    #     for row_idx, (boxes, confs, clses) in enumerate(preds):
    #         image_bgr = images_bgr[row_idx]
    #         if len(boxes) != 0:
    #             for xyxy, conf, cls in zip(boxes, confs, clses):
    #                 plot_one_box(xyxy, image_bgr, label=cls, color=(255, 0, 0))
    #                 print(xyxy, conf, cls)

    #         cv2.imshow('img', image_bgr) 
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()


# class TestYolov4DetectorVideo(unittest.TestCase):
#     def test_detector_yolov4_tiny(self):
#         cfg_fp, data_fp, weights_fp = io.get_params('yolov4_tiny')
#         detector = Detector(cfg_fp, data_fp, weights_fp)
#         img_fp = io.get_test_data('road_video')

#         cap = cv2.VideoCapture(img_fp)
#         count = 0
#         st = datetime.now()
#         while(True):
#             ret, image_bgr = cap.read()

#             conf_thres = 0.15
#             iou_thres = 0.6
#             boxes, confs, clses = detector.detect(image_bgr, conf_thres=conf_thres, iou_thres=iou_thres)
#             if boxes is not None:
#                 for xyxy, conf, cls in zip(boxes, confs, clses):
#                     plot_one_box(xyxy, image_bgr, label=cls, color=(255, 0, 0))


#             cv2.imshow('frame', image_bgr)
#             count += 1
#             if datetime.now()- st > timedelta(seconds=1):
#                 print("fps:", count)
#                 count = 0
#                 st = datetime.now()

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         # When everything done, release the capture
#         cap.release()
#         cv2.destroyAllWindows()


if __name__ == "__main__":
    unittest.main()
#  python -m unittest -v test.py
