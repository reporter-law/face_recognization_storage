"""程序说明"""
# -*-  coding: utf-8 -*-
# Author: cao wang
# Datetime : 2020
# software: PyCharm
# 收获:
from moviepy.video.io.VideoFileClip import VideoFileClip
import os


def GetVideo():

    vedio2 = VideoFileClip(r"F:\IDM下载内容保存\you-get\呆妹儿小霸王.mp4")
    toalvideo = vedio2.set_audio("F:\IDM下载内容保存\you-get\【Douyu】呆妹儿小霸王.m4a")
    toalvideo.write_videofile(r"‪J:\vedio\daimei——hecheng.mp4")


def video_change_Movie(input_vedio,output_audio):
    os.system("ffmpeg -i {input_vedio} -vn -codec copy {output_audio}".format(input_vedio=input_vedio,
                                                                              output_audio=output_audio))
def movie(output_audio,input_mp4,output_mp4):

    os.system("ffmpeg -i {output_audio} -i {input_mp4} -c copy {output_mp4}".format(output_audio=output_audio,input_mp4=input_mp4,output_mp4=output_mp4))
input_vedio=r"J:\vedio\孙小果再次受审并做陈述.flv".strip('\u202a').strip()#原vedio
output_audio="J:\\vedio\\manager\\audio\\孙小果再次受审.m4a".strip('\u202a').strip()#提取的音频
input_mp4= r"J:\\vedio\\manager\\孙小果案.avi".strip('\u202a').strip()#识别的视频
output_mp4=r"J:\\vedio\\manager\\movie\\孙小果受审.avi".strip('\u202a').strip()#最终输出q
#video_change_Movie(input_mp4,output_audio)
movie(output_audio,input_mp4,output_mp4)

def intercept_movie():

    os.system("ffmpeg -ss 951 -t 330 -i F:\IDM下载内容保存\you-get\美国罪案故事.American.Crime.Story.S01E05.中英字幕.BD-HR.AAC.720p.x264-人人影视.mp4 -codec copy F:\IDM下载内容保存\you-get\美国罪案故事.mp4" )
#intercept_movie()