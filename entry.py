from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from collections import deque
from detector import finder
#from CorrelationTracker import Tracker
import dlib

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", type=str,
	default="MobileNetSSD_deploy.prototxt.txt", help="path to Caffe 'deploy' prototxt file" )
ap.add_argument("-m", "--model", type=str,
	default="MobileNetSSD_deploy.caffemodel", help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer")
ap.add_argument("-v", "--video", help="path to video")

args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = "person"
COLORS = (0,255,0) 

# load our serialized model from disk
print("[INFO] loading mdel...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

if not args.get("video", False):
    camera = cv2.VideoCapture(0)

else:
    camera = cv2.VideoCapture(args["video"])
# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

##### GLOBAL VARS #####
start_position = 1
frame_index = 0
person_found = False

## 1 => DETECTOR #########
frame, person_coords = finder(args, camera, net, CLASSES, COLORS, fps)
x1,y1,x2,y2 = person_coords

## 2 => INSTANTIATE A TRACKER & PASS IN COORDINATES#######
tracker = dlib.correlation_tracker()
tracker.start_track(frame, dlib.rectangle(x1,y1,x2,y2))

while True:
    retval, fr = camera.read()
    frame_index += 1

    tracker.update(fr)
    p = tracker.get_position()

    point1 = (int(p.left()), int(p.top()))
    point2 = (int(p.right()), int(p.bottom()))

    cv2.rectangle(fr, point1, point2, (0, 0, 255), 2)
    print( ">> TRACKER @--> [{}, {}] \r".format(point1, point2))
    #center coordinates as a tuple
    center = ( int( (point1[0] + point2[0])/2), 
            int( (point1[1] + point2[1])/2)  )
    cv2.circle(fr, (center[0], center[1]), 10, (0,255,0), 2)

    cv2.namedWindow("TRACKED", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("TRACKED", fr)

    # if the `q` key was pressed, break from the loop
    key = cv2.waitKey(50) & 0xFF
    if key == ord("q"):
            break


# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
camera.release()
cv2.destroyAllWindows()
