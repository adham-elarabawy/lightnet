from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import argparse


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--out", dest='output', help="video path to store the video in(include the desired video name and type[avi])",
                        default="output.avi", type=str),
    parser.add_argument("--src", dest='source', help="path of the video that the model is being tested on(include the desired video name and type)",
                        default="input.avi", type=str),
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence",
                        help="Object Confidence to filter predictions", default=0.25)
    parser.add_argument("--nms_thresh", dest="nms_thresh",
                        help="NMS Threshhold", default=0.45),
    parser.add_argument("--cfg", dest='cfg', help="Config file",
                        default="cfg/yolov3.cfg", type=str),
    parser.add_argument("--weights", dest='weights', help="weightsfile",
                        default="yolov3.weights", type=str),
    parser.add_argument("--meta", dest='data', help="path to the .data file detailing the model metadata",
                        default="cfg/model.data", type=str)
    parser.add_argument('--show', dest='show', action='store_true')
    parser.add_argument('--dont_show', dest='show', action='store_false')
    parser.set_defaults(feature=False)
    return parser.parse_args()


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
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


netMain = None
metaMain = None
altNames = None


def YOLO(args):

    global metaMain, netMain, altNames
    configPath = args.cfg
    weightPath = args.weights
    metaPath = args.data

    print("+*Using:\n\tCFG: " + configPath + "\n\tWeights: " +
          weightPath + "\n\tMetadata: " + metaPath)
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, args.bs)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(args.source)
    cap.set(3, 1920)
    cap.set(4, 1080)
    num_frames = cap.get(7)
    out = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                       darknet.network_height(netMain), 3)

    currFrame = 0
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break
        if(ret):
            currFrame += 1
            frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb,
                                       (darknet.network_width(netMain),
                                        darknet.network_height(netMain)),
                                       interpolation=cv2.INTER_LINEAR)

            darknet.copy_image_from_bytes(
                darknet_image, frame_resized.tobytes())

            detections = darknet.detect_image(
                netMain, metaMain, darknet_image, thresh=args.confidence, nms=args.nms_thresh, debug=False)
            image = cvDrawBoxes(detections, frame_resized)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            out.write(image)
            print("fps: " + str(int(1/(time.time()-prev_time))))
            if(args.show):
                cv2.imshow('Demo', image)
                cv2.waitKey(3)
            if(currFrame == num_frames):
                print("Successfully finished and exported to: " + args.output)
                break
    cap.release()
    out.release()


if __name__ == "__main__":
    args = arg_parse()
    YOLO(args)
