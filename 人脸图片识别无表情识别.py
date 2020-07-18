"""程序说明"""
# -*-  coding: utf-8 -*-
# Author: cao wang
# Datetime : 2020
# software: PyCharm
# 收获:
from imutils import face_utils
import dlib
import imutils
import cv2


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("J:\\PyCharm项目\\项目\\庭审微表情识别\\face_classification\\shape_predictor_68_face_landmarks.dat")

image = cv2.imread("F:\\Pictures\\司考\\096A4294司考.jpg")
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)

# enumerate()方法用于将一个可遍历的数据对象(列表、元组、字典)组合
# 为一个索引序列，同时列出 数据下标 和 数据 ，一般用在for循环中
for(i, rect) in enumerate(rects):
    shape = predictor(gray, rect)  # 标记人脸中的68个landmark点
    shape = face_utils.shape_to_np(shape)  # shape转换成68个坐标点矩阵

    (x, y, w, h) = face_utils.rect_to_bb(rect)  # 返回人脸框的左上角坐标和矩形框的尺寸
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    landmarksNum = 0;
    for (x, y) in shape:
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        # cv2.putText(image, "{}".format(landmarksNum), (x, y),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1)
        # landmarksNum = landmarksNum + 1;
    landmarksNum = 0;
cv2.imshow("Output", image)

cv2.waitKey(0)

