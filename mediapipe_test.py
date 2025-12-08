import mediapipe as mp
import cv2
import csv
import numpy as np
import os 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

labels = ['x','y','z']
dataFile = "data.csv"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE
)

with HandLandmarker.create_from_options(options) as landmarker:

    image = mp.Image.create_from_file("100418777_H_1.jpg")
    result = landmarker.detect(image)

    annotated_image = image.numpy_view().copy()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    
    #Check if csv exists
    if not os.path.exists(dataFile):
        with open(dataFile, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(labels)
            
    #Check if labels are currently written 
    else:
        with open(dataFile,'r') as f:
            reader = csv.reader(f)
            row1 = next(reader,None)
            
            print(row1)
            
        if row1 is not None and [col.strip().lower() for col in row1] == labels:
            print("already printed")
             
        else:
            with open(dataFile,'a', newline="\n") as f:
                writer = csv.writer(f)
                writer.writerow(labels)

    # Extract x, y, z for each landmark
    for hand_landmarks in result.hand_landmarks:
         xyz_list = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]


    #Writing x,y,z data to csv   
    with open(dataFile,'a', newline="") as f:
        writer = csv.writer(f)
        for (x,y,z) in xyz_list:
            writer.writerow([x,y,z]) 
        writer.writerow("")          

        landmark_list = landmark_pb2.NormalizedLandmarkList(
            landmark=[
                landmark_pb2.NormalizedLandmark(
                    x=lm.x, y=lm.y, z=lm.z
                ) for lm in hand_landmarks
            ]
        )

        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            landmark_list,
            mp.solutions.hands.HAND_CONNECTIONS
        )

    cv2.imshow("Annotated", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
