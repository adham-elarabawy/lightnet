import os
import cv2
import ntpath
import sys

image_path_list = []
source = "/Users/adhamelarabawy/Documents/GitHub/Yolo_mark/x64/Release/data/t3/img_full"

for file in os.listdir(source):
    image_path_list.append(os.path.join(source, file))

# loop through image_path_list to open each image
for imagePath in image_path_list:
    if imagePath.endswith(".jpg") and (not imagePath.endswith("Y.jpg")) and (not imagePath.endswith("Y2.jpg")) and (not imagePath.endswith("Y3.jpg")):
        img_in = cv2.imread(imagePath)
        img_yuv = cv2.cvtColor(img_in, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(img_yuv)
        y2 = (y/2) + 64
        y3 = (y/3) + 85
        print(imagePath, end='\r')
        sys.stdout.flush()
        cv2.imwrite(imagePath[:-4] + "Y.jpg", y)
        cv2.imwrite(imagePath[:-4] + "Y2.jpg", y2)
        cv2.imwrite(imagePath[:-4] + "Y3.jpg", y3)
