"""程序说明"""
# -*-  coding: utf-8 -*-
# Author: cao wang
# Datetime : 2020
# software: PyCharm
# 收获:
import cv2  # 图像处理的库 OpenCv
import dlib  # 人脸识别的库 dlib
import numpy as np  # 数据处理的库 numpy


# class
class face_emotion():

    def __init__(self):
        # 使用特征提取器 get_frontal_face_detector
        self.detector = dlib.get_frontal_face_detector()
        # dlib 的68点模型，使用官方训练好的特征预测器
        self.predictor = dlib.shape_predictor("J:\\PyCharm项目\\项目\\庭审微表情识别\\face_classification\shape_predictor_68_face_landmarks.dat")

        # 建cv2摄像头对象，参数0表示打开电脑自带的摄像头，如果换了外部摄像头，则自动切换到外部摄像头
        self.cap = cv2.VideoCapture(r"F:\you-get\在死刑边缘疯狂试探，狂怼审判长：“你有本事拿枪来打死我啊？”.flv")
        # set(propId, value),设置视频参数，propId设置视频参数， value设置参数值
        self.cap.set(3, 480)
        # 截取 screenshoot 的计数器
        self.cnt = 0

        """保存"""
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.outVideo = cv2.VideoWriter(r"J:\vedio\manager\68_face_test.avi".strip('\u202a'),
                                   cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, self.size)

    def learning_face(self):
        # 眉毛直线拟合数据缓冲
        line_brow_x = []
        line_brow_y = []

        # 返回true/false 检查初始化是否成功
        # cap.isOpened()
        while (self.cap.isOpened()):

            # cap.read()，按帧读取视频，ret,frame是其两个返回值
            #  ret，布尔值 true/false，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。
            #  frame，每一帧的图像对象，图像的三维矩阵。
            flag, im_rd = self.cap.read()
            # cv2.waitKey()
            # 参数是1，表示延时1ms切换到下一帧图像，对于视频而言；
            # 参数为0，表示只显示当前帧图像，相当于视频暂停；
            k = cv2.waitKey(1)
            # 取灰度
            img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)

            # 使用人脸检测器检测每一帧图像中的人脸，并返回人脸数 faces
            faces = self.detector(img_gray, 0)

            # 待会要显示在屏幕上的字体
            font = cv2.FONT_HERSHEY_SIMPLEX

            # 如果检测到人脸
            if (len(faces) != 0):

                # 对每个人脸都标出68个特征点
                for i in range(len(faces)):
                    # enumerate 方法同时返回数据对象的索引和数据，k为索引，d为faces中的对象
                    for k, d in enumerate(faces):
                        # 用红色矩阵框出人脸, 光的三原色Red(0,0,255), Green(0,255,0), Blue(255,0,0)
                        #  rectangle(img, pt1, pt2, color), 其中pt1为矩阵上顶点，pt2为矩阵下顶点
                        cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255))
                        # 计算人脸识别框边长
                        self.face_width = d.right() - d.left()

                        # 使用预测器得到68点数据的坐标
                        shape = self.predictor(im_rd, d)

                        # 圆圈显示每个特征点
                        for i in range(68):
                            # circle(img, center, radius, color),
                            # img,Image where the circle is drawn
                            # center,Center of the circle
                            # radius,Radius of the circle (半径)
                            # color，Circle color
                            cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 5, (0, 255, 0), -1, 8)
                            # putText(img, text, org, fontFace, fontScale, color)
                            # org :Bottom-left corner of the text string in the image.
                            cv2.putText(im_rd, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (255, 255, 255))

                        # 分析任意 n 点的位置关系来作为表情识别的依据
                        # 嘴中心	66，嘴左角48，嘴右角54
                        mouth_width = (shape.part(54).x - shape.part(48).x) / self.face_width  # 嘴巴张开程度
                        mouth_height = (shape.part(66).y - shape.part(62).y) / self.face_width  # 嘴巴张开程度
                        # print("嘴巴宽度与识别框宽度之比：" , mouth_width)
                        # print("嘴巴高度与识别框宽度之比：" , mouth_height)

                        # 通过两个眉毛上的10个特征点，分析挑眉程度和皱眉程度
                        brow_sum = 0  # 高度之和
                        frown_sum = 0  # 两边眉毛距离之和
                        for j in range(17, 21):
                            brow_sum += (shape.part(j).y - d.top()) + (shape.part(j + 5).y - d.top())
                            frown_sum += shape.part(j + 5).x - shape.part(j).x
                            line_brow_x.append(shape.part(j).x)
                            line_brow_y.append(shape.part(j).y)

                        # self.brow_k, self.brow_d = self.fit_slr(line_brow_x, line_brow_y) # 计算眉毛的倾斜程度
                        tempx = np.array(line_brow_x)
                        tempy = np.array(line_brow_y)
                        z1 = np.polyfit(tempx, tempy, 1)  # 拟合成一次直线
                        self.brow_k = -round(z1[0], 3)  # 拟合出曲线的斜率和实际眉毛的倾斜方向是相反的

                        brow_height = (brow_sum / 10) / self.face_width  # 眉毛高度占比
                        brow_width = (frown_sum / 5) / self.face_width  # 眉毛距离占比
                        # print("眉毛高度与识别框宽度之比：" , brow_height)
                        # print("眉毛间距与识别框高度之比：" , brow_width)

                        # 眼睛睁开程度
                        eye_sum = (shape.part(41).y - shape.part(37).y + shape.part(40).y - shape.part(38).y +
                                   shape.part(47).y - shape.part(43).y + shape.part(46).y - shape.part(44).y)
                        eye_hight = (eye_sum / 4) / self.face_width
                        # print("眼睛睁开距离与识别框高度之比：" , eye_hight)

                        # 分情况讨论,判断情绪变化
                        # 张嘴，可能是开心或惊讶，通过眼睛的睁开程度区分
                        if round(mouth_height >= 0.03):
                            if eye_hight >= 0.056:
                                cv2.putText(im_rd, "amazing", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2, 4)
                            else:
                                cv2.putText(im_rd, "happy", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2, 4)

                        # 没有张嘴，可能是正常和生气，通过眉毛区分
                        else:
                            if self.brow_k <= -0.3:
                                cv2.putText(im_rd, "angry", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2, 4)
                            else:
                                cv2.putText(im_rd, "nature", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2, 4)
                # 标出人脸数
                cv2.putText(im_rd, "Face-" + str(len(faces)), (20, 50), font, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                # 没有检测到人脸
                cv2.putText(im_rd, "No Face", (20, 50), font, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

            # 添加说明
            im_rd = cv2.putText(im_rd, "S: screenshot", (20, 450), font, 0.6, (255, 0, 255), 1, cv2.LINE_AA)
            im_rd = cv2.putText(im_rd, "Q: quit", (20, 470), font, 0.6, (255, 0, 255), 1, cv2.LINE_AA)

            # 按下 s 键截图保存
            # waitKey()函数的功能是不断刷新图像，频率时间为delay，单位为ms
            if (cv2.waitKey(1) & 0xFF) == ord('s'):
                self.cnt += 1
                cv2.imwrite("screenshoot" + str(self.cnt) + ".jpg", im_rd)

            # 按下 q 键退出
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

            # 窗口显示
            im_rd = cv2.resize(im_rd, dsize=self.size)
            cv2.imshow("Face Recognition", im_rd)
            self.outVideo.write(im_rd)

        # 释放摄像头
        self.cap.release()
        # 删除建立的窗口
        cv2.destroyAllWindows()


# main
if __name__ == "__main__":
    my_face = face_emotion()
    my_face.learning_face()
