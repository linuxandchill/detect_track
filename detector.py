import numpy as np
import imutils
import cv2
import Person
from collections import deque

def finder(args, camera, net, CLASSES, COLORS, fps):
    people = []
    person_found = False
    id = 1

    while True:
    # loop over the frames from the video stream
        #resize frame captured from thread
        grabbed, frame = camera.read()
        #print("Found: {}".format(person_found))

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
            #print("Length: {}, type: {}".format(len(people), type(people)))

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                height = abs(endY - startY)
                width = abs(endX - startX)
                print(height, width)
                topLeft = (startX, startY)
                bottomRight = (endX, endY)
                
                new = True
                for p in people:
                    #close to old object
                    if abs(topLeft[0]-p.tl[0])<=width and abs(topLeft[1]-p.tl[1])<=height: 
                        new = False
                        p.updateCoords(topLeft, bottomRight)
                        break

                if new == True:
                #create new person
                    #use coords to create a Person object
                    #add to 'people' deque
                    #increment ID
                    new_person = Person.Person(topLeft, bottomRight, id)
                    people.append(new_person)
                    id += 1
                    first_person_tl, first_person_br = people[0].tl, people[0].br #ret (x1,y1), (x2,y2)
                
                print("FIRST PERSON {}, {}, ID: {}".format(first_person_tl, first_person_br, people[0].id))

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES,
                        confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (0,255,0), 2)
                #cv2.circle(frame, (center_coords[0], center_coords[1]), 10, (255,0,255), 8)

                #label position
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                
                #update person_found status
                person_found = True 
        
        #print(len(people))
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if person_found or key==ord('q'):
            break
       
        # update the FPS counter
        fps.update()
    #handle this exception if no person found
    return frame, [first_person_tl[0], first_person_tl[1], 
            first_person_br[0], first_person_br[1]]
    """
    return frame, [first_person_tl[0], first_person_tl[1], 
            first_person_br[0], first_person_br[1]]
    """



