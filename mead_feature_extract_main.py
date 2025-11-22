from mediapipe.tasks.python import vision
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import opensmile
import pandas

def get_features(image_dir, audio_dir):
    # Grab feature array from facial landmarks using MediaPipe
    base = mp.tasks.BaseOptions(model_asset_path="face_landmarker.task")
    options = vision.FaceLandmarkerOptions(base_options=base,
                                        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    image = mp.Image.create_from_file(image_dir)

    result = detector.detect(image)

    face_features = np.array([[landmark.x, landmark.y, landmark.z] for face in result.face_landmarks for landmark in face]).flatten()

    #Grab feature array from audio using OpenSMILE
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )

    audio_features = smile.process_file(audio_dir).to_numpy().flatten()

    #Concatenate
    final_array = np.concatenate((face_features, audio_features))

    return final_array


print(get_features("test_faces/face.jpg", "test_audio/audio1.wav"))
