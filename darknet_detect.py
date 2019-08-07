from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time as _time
import darknet
import argparse
import sys
import filetype
from pyzbar.pyzbar import decode as dcdB

DEBUG_PRINT = True  # set to True to enable all debug prints and output paths

supportedVideoFormats = ['mkv', 'avi', 'mov', 'mp4']
supportedImageFormats = ['png', 'jpg', 'jpeg', 'bmp']
validBarcodesList = []
profile = [0, 0, 0, 0, 0]
tempPrev = _time.time()

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]


def arg_parse():
    """
    Parse arguments to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument('--out', dest='output', help='**images & vids only** path to store the output in(include the desired filename and filetype) // video:.avi, image:.png, directory:directoryname',
                        default='___', type=str),
    parser.add_argument('--src', dest='source', help='path of the file/directory(of images) that the model is being tested on(include the filename and filetype) [SET TO 0 FOR WEBCAM USE(not tested)] \n *If passing in directory, ALL files in directory MUST BE valid images',
                        required=True, type=str),
    parser.add_argument('--bs', dest='bs', help='Batch size', default=1),
    # parser.add_argument('--confidence', dest='confidence',
    #                     help='Object Confidence to filter predictions', default=0.25),
    # parser.add_argument('--nms_thresh', dest='nms_thresh',
    #                     help='NMS Threshhold', default=0.45),
    parser.add_argument('--cfg', dest='cfg', help='Config file',
                        required=True, type=str),
    parser.add_argument('--weights', dest='weights', help='weightsfile',
                        required=True, type=str),
    parser.add_argument('--meta', dest='data', help='path to the .data file detailing the model metadata',
                        required=True, type=str),
    parser.add_argument('--outfps', dest='fps', help='desired framerate of the output video(with the bounding boxes on it)[LIMITED BY PROCESSING SPEED]',
                        default=30, type=int),
    parser.add_argument('--len', dest='displayLength',
                        help='(FOR IMAGE DETECTION NOT VIDEO), how long to display the processed frame before ending the program',
                        default=10000, type=int),
    parser.add_argument('--show', dest='show', action='store_true',
                        help='show the frames as they are being processed(LOWERS PERFORMANCE SIGNIFICANTLY)'),
    parser.add_argument('--resize', dest='resize', action='store_true',
                        help='resize processed frames to dimensions of the neural network')
    parser.set_defaults(show=False)
    parser.set_defaults(resize=False)
    parser.set_defaults(cam=False)

    return parser.parse_args()


def checkType(source):
    if os.path.isdir(source):
        return 2  # input is a directory
    kind = filetype.guess(source)
    if os.path.isfile(source):
        if kind is None:
            return -1  # input is not a supported type/format
        for videoType in supportedVideoFormats:
            if kind.extension == videoType:
                return 0  # input is supported VIDEO format
        for imageType in supportedImageFormats:
            if kind.extension == imageType:
                return 1  # input is supported IMAGE format
    else:
        print(source + ' is not a file or directory: Quitting...')
        sys.exit()


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    available_colors = len(colors)
    tempI = 0
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, colors[tempI], 3)
        cv2.putText(img,
                    detection[0].decode() +
                    ' [' + str(round(detection[1] * 100, 2)) + ']',
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                    colors[tempI], 1)
        tempI += 1
        if(tempI > available_colors - 1):
            tempI = 0
    return img


validBarcodesList = []


def cropToBoundingBox(detections, img, args, imagePath):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        x1 = max(0, round(x - w/2))
        y1 = max(0, round(y - h/2))
        x2 = max(0, round(x + w/2))
        y2 = max(0, round(y + h/2))
        crop_img = img[y1:y2, x1:x2]
        if(args.show):
            cv2.imshow('demo', crop_img)
            cv2.waitKey(3)
        height, width, channels = img.shape
        frame_resized = cv2.resize(
            img, (width*args.scale, height*args.scale), interpolation=cv2.INTER_LANCZOS4)
        decodedInfo = dcdB(frame_resized)
        if len(decodedInfo) != 0:
            validBarcodesList.append(imagePath)


def midLineBarcodeCrop(detections, img, args, imagePath):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]

        x1 = max(0, round(x - w/2))
        y1 = max(0, round(y+1))
        x2 = max(0, round(x + w/2))
        y2 = max(0, round(y - 1))
        crop_img = img[y1:y2, x1:x2]
        if(args.show):
            cv2.imshow('demo', crop_img)
            cv2.waitKey(3)
        height, width, channels = img.shape
        # frame_resized = cv2.resize(
        #     img, (width*args.scale, height*args.scale), interpolation=cv2.INTER_LANCZOS4)
        decodedInfo = dcdB(frame_resized)
        if len(decodedInfo) != 0:
            validBarcodesList.append(imagePath)


def processFrame(frameToProcess, args, darknet_image, netMain, tempPrev):
    # THIS IS REQUIRED BECAUSE OPENCV SWITCHES R & B CHANNELS!
    frameToProcess = cv2.cvtColor(frameToProcess, cv2.COLOR_BGR2RGB)
    darknet.copy_image_from_bytes(
        darknet_image, frameToProcess.tobytes())
    profile[1] = profile[1] + (_time.time() - tempPrev)
    tempPrev = _time.time()
    detections = darknet.detect_image(
        netMain, metaMain, darknet_image, debug=False)
    profile[2] = profile[2] + (_time.time() - tempPrev)
    tempPrev = _time.time()
    # draw bounding boxes on the processed frame
    markedImage = cvDrawBoxes(detections, frameToProcess)
    profile[3] = profile[3] + (_time.time() - tempPrev)
    # convert colorspace back to rgb from opencv native
    return cv2.cvtColor(markedImage, cv2.COLOR_BGR2RGB)


def resizeMaintain(frameToProcess, netMain):
    # frame_resized = cv2.resize(frameToProcess, (darknet.network_width(
    #     netMain), darknet.network_height(netMain)), interpolation=cv2.INTER_LINEAR)
    return frameToProcess
    # RESIZE WITH PADDING:
    # desired_size = 1920  # darknet.network_width(netMain)
    # # old_size is in (height, width) format
    # old_size = frameToProcess.shape[:2]

    # ratio = float(desired_size)/max(old_size)
    # new_size = tuple([int(x*ratio) for x in old_size])

    # # new_size should be in (width, height) format

    # im = cv2.resize(frameToProcess, (new_size[1], new_size[0]))

    # delta_w = desired_size - new_size[1]
    # delta_h = desired_size - new_size[0]
    # top, bottom = delta_h//2, delta_h-(delta_h//2)
    # left, right = delta_w//2, delta_w-(delta_w//2)

    # color = [0, 0, 0]
    # new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
    #                             value=color)
    # return new_im


netMain = None
metaMain = None
altNames = None


def YOLO(args):

    global metaMain, netMain, altNames
    configPath = args.cfg
    weightPath = args.weights
    metaPath = args.data

    if DEBUG_PRINT:
        print('+*Using:\n\tCFG: ' + configPath + '\n\tWeights: ' +
              weightPath + '\n\tMetadata: ' + metaPath)

    if not os.path.exists(configPath):
        raise ValueError('Invalid config path `' +
                         os.path.abspath(configPath)+'`')
    if not os.path.exists(weightPath):
        raise ValueError('Invalid weight path `' +
                         os.path.abspath(weightPath)+'`')
    if not os.path.exists(metaPath):
        raise ValueError('Invalid data file path `' +
                         os.path.abspath(metaPath)+'`')
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            'ascii'), weightPath.encode('ascii'), 0, args.bs)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode('ascii'))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search('names *= *(.*)$', metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split('\n')
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    fileType = 3
    output = 'liveVideo_proc.avi'
    if not args.source == "0":
        fileType = checkType(args.source)
        if not fileType == 2:
            # file type of the input image
            inputType = filetype.guess(args.source)
            output = args.output
            if output == '___':
                toStrip = len(str(inputType.extension)) + 1
                output = args.source[:-toStrip] + \
                    '_proc.' + str(inputType.extension)

    if fileType == -1:
        print('Input is not a supported image or video format. Try running the python script with: --help for more information')
        sys.exit()

    if fileType == 0:  # input is a video
        if DEBUG_PRINT:
            print('Validated: Source input is a video.')
        cap = cv2.VideoCapture(args.source)
        cap.set(3, 1920)
        cap.set(4, 1080)
        num_frames = cap.get(7)

        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        print('Starting the YOLO loop...')

        currFrame = 0
        while True:
            prev_time = _time.time()
            ret, frame_read = cap.read()
            if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
                break
            if(ret):
                if currFrame == 0:
                    height, width, channels = frame_read.shape
                    out = cv2.VideoWriter(
                        output, cv2.VideoWriter_fourcc(*'MJPG'), args.fps,
                        (width, height))
                    # create an image we reuse for each detect
                    darknet_image = darknet.make_image(width, height, channels)
                profile[0] = profile[0] + (_time.time() - prev_time)
                tempPrev = _time.time()
                currFrame += 1
                processedFrame = processFrame(
                    frame_read, args, darknet_image, netMain, tempPrev)
                tempPrev = _time.time()
                # add processed frame to the output file
                out.write(processedFrame)
                profile[4] = profile[4] + (_time.time() - tempPrev)
                print('avg inference fps: ' + str(int(currFrame/(profile[2]))) + ', actual fps: ' + str(int(1/(_time.time()-prev_time))) +
                      ', frames processed: ' + str(currFrame) + '/' + str(num_frames), end='\r')
                sys.stdout.flush()
                if(args.show):
                    cv2.imshow('Demo', processedFrame)
                if(currFrame == num_frames):
                    print('Successfully finished and exported to: ' + output)
                    break
        cap.release()
        out.release()
        # index = 0
        # for time in profile:
        #     profile[index] = time/num_frames
        #     index += 1
        print(profile)

    if fileType == 1:  # input is an image
        if DEBUG_PRINT:
            print('Validated: Source input is an image.')
        frame_read = cv2.imread(args.source)
        print('Starting the YOLO loop...')
        height, width, channels = frame_read.shape
        # create an image we reuse for each detect
        if(args.resize):
            height = darknet.network_height(netMain)
            width = darknet.network_width(netMain)
        darknet_image = darknet.make_image(width, height, channels)
        processedFrame = processFrame(
            frame_read, args, darknet_image, netMain)
        if(args.show):
            cv2.imshow('Demo', processedFrame)
            cv2.waitKey(args.displayLength)
        cv2.imwrite(output, processedFrame)
        print('Successfully finished and exported to: ' + output)

    if filetype == 2:  # input is a directory
        if DEBUG_PRINT:
            print('Validated: Source input is a directory.')
        image_path_list = []
        print('Populating list with all files in directory...')
        for file in os.listdir(args.source):
            image_path_list.append(os.path.join(args.source, file))
        print('Starting the YOLO loop...')
        # loop through image_path_list to open each image
        for imagePath in image_path_list:
            # determine output path for processed frame
            inputType = filetype.guess(imagePath)
            toStrip = len(str(inputType.extension)) + 1
            output = imagePath[:-toStrip] + '_proc.' + str(inputType.extension)

            frame_read = cv2.imread(args.source)
            height, width, channels = frame_read.shape
            # create an image we reuse for each detect
            if(args.resize):
                height = darknet.network_height(netMain)
                width = darknet.network_width(netMain)
            darknet_image = darknet.make_image(width, height, channels)
            processedFrame = processFrame(
                frame_read, args, darknet_image, netMain)
            if(args.show):
                cv2.imshow('Demo', processedFrame)
                cv2.waitKey(args.displayLength)
            cv2.imwrite(output, processedFrame)
            print('Successfully finished frame and exported to: ' +
                  output + '\nMoving on to next frame...', end='\r')
    if fileType == 3:  # input is camera stream
        if DEBUG_PRINT:
            print('Validated: Source input is a camera stream.')
        cap = cv2.VideoCapture(0)
        opened = cap.open(0)
        print(str(opened))

        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        print('Starting the YOLO loop...')

        currFrame = 0
        while True:
            prev_time = _time.time()
            ret, frame_read = cap.read()
            if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
                break
            if(ret):
                if currFrame == 0:
                    height, width, channels = frame_read.shape
                    out = cv2.VideoWriter(
                        output, cv2.VideoWriter_fourcc(*'MJPG'), args.fps,
                        (width, height))
                    # create an image we reuse for each detect
                    darknet_image = darknet.make_image(width, height, channels)
                profile[0] = profile[0] + (_time.time() - prev_time)
                tempPrev = _time.time()
                currFrame += 1
                processedFrame = processFrame(
                    frame_read, args, darknet_image, netMain, tempPrev)
                tempPrev = _time.time()
                # add processed frame to the output file
                out.write(processedFrame)
                profile[4] = profile[4] + (_time.time() - tempPrev)
                print('avg inference fps: ' + str(int(currFrame/(profile[2]))) + ', actual fps: ' + str(int(1/(_time.time()-prev_time))) +
                      ', frames processed: ' + str(currFrame), end='\r')
                sys.stdout.flush()
                if(args.show):
                    cv2.imshow('Demo', processedFrame)
        cap.release()
        out.release()
        print(profile)


if __name__ == '__main__':
    args = arg_parse()
    YOLO(args)
    cv2.destroyAllWindows()
