import os
import cv2
import ntpath
import sys

image_path_list = []
source = "/Users/adhamelarabawy/Documents/temp/test/"

for file in os.listdir(source):
    image_path_list.append(os.path.join(source, file))

# loop through image_path_list to open each image
for imagePath in image_path_list:
    if imagePath.endswith(".jpeg") and (not imagePath.endswith("Y.jpg")) and (not imagePath.endswith("Y2.jpg")) and (not imagePath.endswith("Y3.jpg")):
        img_in = cv2.imread(imagePath)
        b, g, r = cv2.split(img_in)
        print(str(b) + ',' + str(g) + ',' + str(r))
        print('--------------------\n\n')
        img_yuv = cv2.cvtColor(img_in, cv2.COLOR_BGR2YCrCb)
        y, u, v = cv2.split(img_yuv)
        print(str(y) + ',' + str(u) + ',' + str(v))
        # img_combined = cv2.merge((g, g, g))
        # cv2.imshow("combined", img_combined)
        # cv2.waitKey(10000)
        # print(str(cv2.split(img_combined)))
        # img_yuv = cv2.cvtColor(img_in,CV_RGBYCrCb)
        # img_bgr_stripped = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        # img_yuv = cv2.cvtColor(img_bgr_stripped, cv2.COLOR_BGR2YUV)
        # y, u, v = cv2.split(img_in)
        # print(str(y) + ',' + str(u) + ',' + str(v))
        # cv2.imshow("y", y)
        # cv2.waitKey(10000)
        # cv2.imshow("u", u)
        # cv2.waitKey(10000)
        # cv2.imshow("v", v)
        # cv2.waitKey(10000)
        # print(str(y) + ',' + str(u) + ',' + str(v))
        # # y2 = (y/2) + 64
        # # y3 = (y/3) + 85
        # print(imagePath)
        # # sys.stdout.flush()
        # cv2.imwrite(imagePath, img_yuv)
        # img_in = cv2.imread(imagePath)
        # y, u, v = cv2.split(img_in)
        # print(str(y) + ',' + str(u) + ',' + str(v))
        # cv2.imwrite(imagePath[:-4] + "Y.jpg", y)
        # cv2.imwrite(imagePath[:-4] + "Y2.jpg", y2)
        # cv2.imwrite(imagePath[:-4] + "Y3.jpg", y3)
