import cv2
import ntpath
import sys
import os.path
from os import path
import ntpath

text_path_list = []
file_path_list = []
file_path_to_remove = []
source = "/Volumes/Extreme SSD/Kelzal/24_class/composite_good/"
train_text_path = "/mnt/nas01/workspace_share/cnn/data/t5/train/"
for file in os.listdir(source):
    file_path_list.append(os.path.join(source, file))

f = open("14_train.txt", "w")

# loop through image_path_list to open each image
for filePath in file_path_list:
    if filePath.endswith(".jpg"):
        text_path_list.append(filePath)
        pathToWrite = train_text_path + ntpath.basename(filePath)
        f.write(pathToWrite + '\n')
        print(pathToWrite)
f.close()
