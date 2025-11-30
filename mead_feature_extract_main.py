from mediapipe.tasks.python import vision
import mediapipe as mp
import numpy as np
import opensmile
import soundfile as sf
import av
from pydub import AudioSegment

base = mp.tasks.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(base_options=base,
                                    num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors
    )

def mead_feature_list(video_path):

    time_series = []
    container = av.open(video_path)

    audio = AudioSegment.from_file(video_path)
    audio.export("test_video/test_audio.wav", format="wav")
    video_sf, sfsr = sf.read("test_video/test_audio.wav")
    audio_duration = len(video_sf) / sfsr
    video_stream = container.streams.video[0]

    audio_window = 1/10
    frame_window = 1/30
    next_t = 0.0
    frame_count = 0
    expected_audio_length = int(audio_window * sfsr)
    
    current_audio_segment = np.zeros(expected_audio_length)

    for frame in container.decode(video_stream):
        t = frame.pts * video_stream.time_base
        
        if t < next_t:
            continue
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame.to_ndarray(format="rgb24")
        )

        result = detector.detect(image)
        face = result.face_landmarks[0]
        face_features = np.array([[lm.x, lm.y, lm.z] for lm in face]).flatten()

        if frame_count % 3 == 0 and t + audio_window <= audio_duration:
            start_sample = int(t * sfsr)
            end_sample = int((t + audio_window) * sfsr)
            segment = video_sf[start_sample:end_sample]

            if len(segment) < expected_audio_length:
                segment = np.pad(segment, (0, expected_audio_length - len(segment)), mode='constant')
            
            current_audio_segment = segment

        voice_features = smile.process_signal(
            current_audio_segment,
            sfsr
        ).to_numpy().flatten()

        next_t += frame_window
        frame_count += 1

        time_step = np.concatenate((face_features, voice_features))
        print(time_step)

        time_series.append(time_step)

    return time_series

print(mead_feature_list("test_video/video_test_first.mp4"))
