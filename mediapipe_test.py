import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE
)

with HandLandmarker.create_from_options(options) as landmarker:

    image = mp.Image.create_from_file("CW2_dataset_final/A/A_sample_1.jpg")
    result = landmarker.detect(image)

    annotated_image = image.numpy_view().copy()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    for hand_landmarks in result.hand_landmarks:
        
        for hand_landmarks in result.hand_landmarks:
        
         # Extract x, y, z for each landmark
         xyz_list = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
         print("Landmark coordinates:")
        for i, (x, y, z) in enumerate(xyz_list):
            print(f"  Landmark {i}: x={x:.4f}, y={y:.4f}, z={z:.4f}")
       

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
