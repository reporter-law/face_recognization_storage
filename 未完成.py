"""程序说明"""
# -*-  coding: utf-8 -*-
# Author: cao wang
# Datetime : 2020
# software: PyCharm
# 收获:

import cv2,dlib,os,glob,numpy,time


# 声明各个资源路径

decter_path =  "J:\PyCharm项目\项目\庭审微表情识别\\face_classification\\trained_models\\shape_predictor_68_face_landmarks.dat"

model_path = "J:\PyCharm项目\项目\庭审微表情识别\\face_classification\\dlib_face_recognition_resnet_model_v1.dat"
#img_path = super_path + "pictures"
video_path = "F:\you-get\孙小果再次受审并做陈述 4团伙成员当庭认罪.flv"

# 加载视频
video = cv2.VideoCapture(0)

# 加载模型
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(decter_path)
facerec = dlib.face_recognition_model_v1(model_path)

# 创建窗口
cv2.namedWindow("人脸识别", cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow("人脸识别", 720,540)

descriptors = []
faces = []
# 处理视频，按帧处理
suc,frame = video.read()
flag = True                  # 标记是否是第一次迭代
i = 0                        # 记录当前迭代到的帧位置
while suc:
    if i % 3 == 0:           # 每3帧截取一帧
        # 转为灰度图像处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = detector(gray, 1)        # 检测帧图像中的人脸
        # 处理检测到的每一张人脸
        for k,d in enumerate(dets):
            shape = sp(gray,d)
            # print(d, d.left(), d.right(), d.bottom(), d.top())
            cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)   # 对人脸画框
            face_descriptor = facerec.compute_face_descriptor(frame, shape)      # 提取特征
            v = numpy.array(face_descriptor)
            # 将第一张人脸照片直接保存
            if flag:
                descriptors.append(v)
                faces.append(frame)
                flag = False
            else:
                sign = True                 # 用来标记当前人脸是否为新的
                l = len(descriptors)
                for i in range(l):
                    distance = numpy.linalg.norm(descriptors[i] - v)    # 计算两张脸的距离
                    # 取阈值0.5，距离小于0.5则认为人脸已出现过
                    if distance < 0.5:
                        # print(faces[i].shape)
                        face_gray = cv2.cvtColor(faces[i], cv2.COLOR_BGR2GRAY)
                        # 比较两张人脸的清晰度，保存更清晰的人脸
                        if cv2.Laplacian(gray, cv2.CV_64F).var() > cv2.Laplacian(face_gray, cv2.CV_64F).var():
                            faces[i] = frame
                        sign = False
                        break
                # 如果是新的人脸则保存
                if sign:
                    descriptors.append(v)
                    faces.append(frame)
        cv2.imshow("人脸识别", frame)      # 在窗口中显示

        index = cv2.waitKey(0)
        if index == 27:
            video.release()
            cv2.destroyWindow("人脸识别")
            break
    suc,frame = video.read()
    i += 1

#print(len(descriptors))     # 输出不同的人脸数

# 将不同的比较清晰的人脸照片输出到本地
j = 1
for fc in faces:
    cv2.imwrite( r"F:\\you-get\孙小果再次受审并做陈述 4团伙成员当庭认罪.flv/result/" + str(j) +".jpg", fc)
    j += 1
