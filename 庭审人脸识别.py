"""程序说明"""
# -*-  coding: utf-8 -*-
# Author: cao wang
# Datetime : 2020
# software: PyCharm
# 收获:
import dlib
import cv2
import numpy as np
import moviepy
from moviepy.video.io.VideoFileClip import VideoFileClip
from keras.models import load_model
from PIL import Image

emotion_mode_path = r"J:\PyCharm项目\项目\庭审微表情识别\FaceEmotion_ID\models\_mini_XCEPTION.102-0.66.hdf5".strip('\u202a')
class EmotionDetection():
    def __init__(self,filepath):
        self.detector = dlib.get_frontal_face_detector()#人脸识别
        self.emotion_classfier = load_model(emotion_mode_path,compile=False)#情绪识别
        self.EMOTIONS = ["angry","disgust","scare","happy","sad","surprised","neutral"]#情绪种类
        self.savapath = "F:\IDM下载内容保存\save.mp4"
        self.filepath = filepath
        self.cap = cv2.VideoCapture("F:\IDM下载内容保存\云清.mp4")

    def learning_face(self, IMG_SIZE=64):
        while (self.cap.isOpened()):

            # cap.read()
            # 返回两个值：
            #    一个布尔值true/false，用来判断读取视频是否成功/是否到视频末尾
            #    图像对象，图像的三维矩阵
            # flag, im_rd = self.cap.read()
            im_rd = self.cap.read()[1]

            if im_rd is None:
                break

            # 取灰度
            img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)


            # 使用人脸检测器检测每一帧图像中的人脸。并返回人脸数rects
            # faces = self.detector(img_gray, 0)
            faces = self.detector(img_gray, 0)

            im_rdclone = im_rd.copy()


            # 如果检测到人脸
            if (len(faces) != 0):
                print(faces)
                Image.open(faces)

                predx=[]
                faces_ = [[i.left(),i.top(),i.right()-i.left(),i.bottom()-i.top()] for i in faces]
                print(faces_)

                faces_ = sorted(faces_,reverse=True,key=lambda x:(x[2]-x[0])*x[3]-x[1])
                for i in faces_:
                    img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (200, 200))
                    predx.append(img)

                predx=np.array(predx)/255.0
                print(predx)
                preds = self.emotion_classfier.predict(predx)


                label = [self.EMOTIONS[i.argmax()] for i in preds]
                for i ,emotion in enumerate(label):
                    cv2.putText(im_rdclone,emotion,(faces_[i][0],faces_[i][1]-10))
        # 释放摄像头
        self.cap.release()

        # 删除建立的窗口
        cv2.destroyAllWindows()

EmotionDetection(r"J:\vedio\daimei.mp4").learning_face()







