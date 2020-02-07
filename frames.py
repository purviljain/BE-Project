# Converts videos into frames at 3fps

import cv2
import os
from pathlib import Path

def getFrame(sec, dir_name):

    cap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    hasFrames, image = cap.read()
    if hasFrames:
        cv2.imwrite("./frames/" + dir_name + "/frame"+ str(count)+ ".jpg", image)
    return hasFrames


for i, video_name in enumerate(os.listdir('./train-video')):
    
    dir_name = video_name.split('.')[0]
    Path('./frames/' + dir_name).mkdir(parents=True, exist_ok=True)
    
    print(video_name)
    cap = cv2.VideoCapture('./train-video/'+ video_name)

    sec = 0
    frameRate = 1/3 
    count = 1
    success = getFrame(sec, dir_name)
    while success:
        
        count += 1
        sec = sec + frameRate
        sec = round(sec, 2)
        if count == 33:
            success = False
        else:
            success = getFrame(sec, dir_name)