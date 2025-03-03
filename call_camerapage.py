from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
# from ui_emotiondetect import Ui_emotiondetect
# from ui_emotiondetect2 import Ui_Form
from ui_emotiondetect3 import Ui_FACELOOK
# import bg_rc
import cv2
from PIL import Image
# from skimage import io
# from skimage.transform import resize
# import os
# import torch
# import pickle
# from torchvision import transforms
# from model import FaceClassifierResNet18
# from model import FaceDetectorResNet34
# from model import EmojiResNet18
# import numpy as np
# from torch.autograd import Variable
# import argparse
# import sys

from explainPart import save_grad_cam_result

class CameraPageWindow(QWidget, Ui_FACELOOK):
    returnSignal = pyqtSignal()

    def __init__(self,parent=None):
        super(CameraPageWindow, self).__init__(parent)
        self.timer_camera = QTimer()  # 初始化定时器
        self.cap = cv2.VideoCapture(0)  # 初始化摄像头
        self.CAM_NUM = 0
        self.setupUi(self)
        self.initUI()
        self.slot_init()
        self.face_emoji = 0
        self.orignal = 0
        self.masked = 0
        self.flag_ = 0
        self.face = None

    def initUI(self):
        # self.setLayout(self.gridLayout)
        self.setupUi(self)

    def getFaceemoji(self,face_emoji, face = None):
        self.face_emoji = face_emoji
        self.face = face

    def slot_init(self):
        # 信号和槽连接
        self.timer_camera.timeout.connect(self.show_camera)
        self.returnButton.clicked.connect(self.exitApp)
        self.cameraButton.clicked.connect(self.slotCameraButton)
        self.explainButton.clicked.connect(self.explain)

    def show_camera(self):
        flag, frame = self.cap.read()
        image_1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.image = Image.fromarray(image_1)
        # print(f"Frame shape: {frame.shape}, RGB shape: {image_1.shape}")
        show = self.face_emoji
        # print("show", show.shape, show.dtype)  # 检查shape和dtype
        show = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)
        showImage = QImage(show.data, show.shape[1], show.shape[0], show.shape[1]*3, QImage.Format_RGB888)
        self.cameraLabel.setPixmap(QPixmap.fromImage(showImage))
        # pass

    def explain(self):
        if self.face is None:  # 检查是否检测到人脸
            print("No face captured")
            pass
        else:
            cv2.imwrite("screenshot.png", self.face)
            save_grad_cam_result('./screenshot.png','./result_screenshot.png')
            pass

    # 打开关闭摄像头控制
    def slotCameraButton(self):

        if self.timer_camera.isActive() == False:
            # 打开摄像头并显示图像信息
            self.openCamera()
            self.flag_ = 1
        else:
            # 关闭摄像头并清空显示信息
            self.closeCamera()
            self.flag_ = 0

    # 打开摄像头
    def openCamera(self):
        self.timer_camera.start(30)
        self.cameraLabel.clear()
        self.cameraButton.setText('关闭摄像头')

    # 关闭摄像头
    def closeCamera(self):
        self.timer_camera.stop()
        self.cameraLabel.clear()
        self.cameraButton.setText('打开摄像头')

    def exitApp(self):
        # 关闭摄像头资源
        if self.timer_camera.isActive():
            self.timer_camera.stop()
        self.cap.release()
        cv2.destroyAllWindows()
        # 发射退出信号
        self.close()  # 关闭窗口

