import cv2
import ntpath
import sys
import os.path
from os import path
import ntpath

text_path_list = []
file_path_list = []
file_path_to_remove = []
source = "/Users/adhamelarabawy/Documents/GitHub/Yolo_mark/x64/Release/data/t3/train"
train_text_path = "/Users/adhamelarabawy/Documents/GitHub/Yolo_mark/x64/Release/data/t3/badboy.txt"
for file in os.listdir(source):
    file_path_list.append(os.path.join(source, file))

# loop through image_path_list to open each image
for filePath in file_path_list:
    if filePath.endswith(".txt"):
        text_path_list.append(filePath)

for textPath in text_path_list:
    if not path.exists(textPath[:-4]+".jpg"):
        file_path_to_remove.append(ntpath.basename(textPath[:-4]+".jpg"))

print(file_path_to_remove)

with open(train_text_path) as oldfile, open(train_text_path[:-4]+"checked.txt", 'w') as newfile:
    for line in oldfile:
        if not any(bad_word in line for bad_word in file_path_to_remove):
            newfile.write(line)
oldfile.close()
newfile.close()
