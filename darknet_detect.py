from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import argparse
import sys
import filetype

DEBUG_PRINT = True  # set to True to enable all debug prints and output paths

supportedVideoFormats = ['mkv', 'avi', 'mov', 'mp4']
supportedImageFormats = ['png', 'jpg', 'jpeg', 'bmp']


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
    parser.add_argument('--confidence', dest='confidence',
                        help='Object Confidence to filter predictions', default=0.25),
    parser.add_argument('--nms_thresh', dest='nms_thresh',
                        help='NMS Threshhold', default=0.45),
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
    return parser.parse_args()


def checkType(source):
    kind = filetype.guess(source)
    if os.path.isdir(source):
        return 2  # input is a directory
    elif os.path.isfile(source):
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
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    ' [' + str(round(detection[1] * 100, 2)) + ']',
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


def processFrame(frameToProcess, args, darknet_image, netMain):
    frame = cv2.cvtColor(
        frameToProcess, cv2.COLOR_BGR2RGB)  # convert to rgb
    if(args.resize):
        frame = cv2.resize(frame, (darknet.network_width(netMain), darknet.network_height(
            netMain)), interpolation=cv2.INTER_LINEAR)  # resize the image to neural network dimensions using interpolation
    darknet.copy_image_from_bytes(
        darknet_image, frame.tobytes())

    detections = darknet.detect_image(
        netMain, metaMain, darknet_image, thresh=args.confidence, nms=args.nms_thresh, debug=False)

    # draw bounding boxes on the processed frame
    markedImage = cvDrawBoxes(detections, frame)

    # convert colorspace back to rgb from opencv native
    return cv2.cvtColor(markedImage, cv2.COLOR_BGR2RGB)


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

    inputType = filetype.guess(args.source)  # file type of the input image

    fileType = checkType(args.source)

    output = args.output
    if output == '___':
        toStrip = len(inputType) + 1
        output = args.source[:-toStrip] + '_proc.' + inputType

    if fileType == -1:
        print('Input is not a supported image or video format. Try running the python script with: --help for more information')
        sys.exit()

    if fileType == 0:  # input is a video
        if DEBUG_PRINT:
            print('Validated: Source input is a video.')
        cap = cv2.VideoCapture(args.source)  # set to 0 to use webcam input
        cap.set(3, 1920)
        cap.set(4, 1080)
        num_frames = cap.get(7)
        print('Starting the YOLO loop...')

        currFrame = 0
        while True:
            prev_time = time.time()
            ret, frame_read = cap.read()
            if currFrame == 0:
                height, width, channels = frame_read.shape
                # create an image we reuse for each detect
                darknet_image = darknet.make_image(width, height, channels)
                out = cv2.VideoWriter(
                    output, cv2.VideoWriter_fourcc(*'MJPG'), args.fps,
                    (width, height))
            if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
                break
            if(ret):
                currFrame += 1
                processedFrame = processFrame(
                    frame_read, args, darknet_image, netMain)
                # add processed frame to the output file
                out.write(processedFrame)
                print('fps: ' + str(int(1/(time.time()-prev_time))), end='\r')
                sys.stdout.flush()
                if(args.show):
                    cv2.imshow('Demo', processedFrame)
                if(currFrame == num_frames):
                    print('Successfully finished and exported to: ' + output)
                    break
        cap.release()
        out.release()

    if fileType == 1:  # input is an image
        if DEBUG_PRINT:
            print('Validated: Source input is an image.')
        frame_read = cv2.imread(args.source)
        print('Starting the YOLO loop...')
        height, width, channels = frame_read.shape
        # create an image we reuse for each detect
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
            toStrip = len(inputType) + 1
            output = imagePath[:-toStrip] + '_proc.' + inputType

            frame_read = cv2.imread(args.source)
            height, width, channels = frame_read.shape
            # create an image we reuse for each detect
            darknet_image = darknet.make_image(width, height, channels)
            processedFrame = processFrame(
                frame_read, args, darknet_image, netMain)
            if(args.show):
                cv2.imshow('Demo', processedFrame)
                cv2.waitKey(args.displayLength)
            cv2.imwrite(output, processedFrame)
            print('Successfully finished frame and exported to: ' +
                  output + '\nMoving on to next frame...', end='\r')


if __name__ == '__main__':
    args = arg_parse()
    YOLO(args)