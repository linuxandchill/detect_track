import numpy as np
import imutils
import cv2
import Person
#from collections import deque

people = []
def finder(args, camera, net, CLASSES, COLORS, fps):
    while True:
    # loop over the frames from the video stream
        #resize frame captured from thread
        grabbed, frame = camera.read()

        if args.get("video") and not grabbed:
            break

        frame = imutils.resize(frame, width=800)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > args["confidence"]:

                #for each detected object, create a points deque
                #center_points = deque(maxlen=args["buffer"])

                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                topLeft = (startX, startY)
                bottomRight = (endX, endY)
                
                people.append([topLeft, bottomRight])
                
                #first person detected
                first_person = Person.Person(people[0][0], people[0][1])
                #print(first_person.tl, first_person.br)

                #center coordinates as a tuple
                #center_coords = ( int( (topLeft[0] + bottomRight[0])/2), 
                #       int( (topLeft[1] + bottomRight[1])/2)  )


                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES,
                        confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (0,255,0), 2)
                #cv2.circle(frame, (center_coords[0], center_coords[1]), 10, (255,0,255), 8)

                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
                break

        # update the FPS counter
        fps.update()
    return frame, [first_person.tl[0], first_person.tl[1], 
            first_person.br[0], first_person.br[1]]

