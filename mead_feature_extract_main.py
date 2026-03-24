from mediapipe.tasks.python import vision
import mediapipe as mp
import numpy as np
import opensmile
import soundfile as sf
import av
import os
from pydub import AudioSegment

MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

base = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
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
    audio_out = video_path.replace(".mp4", "_audio.wav")
    audio.export(audio_out, format="wav")

    video_sf, sfsr = sf.read(audio_out)
    os.remove(audio_out)

    ## logic checks
    print(f"Audio info: shape={video_sf.shape}, sr={sfsr}, duration={len(video_sf)/sfsr:.2f}s")
    if len(video_sf.shape) == 2:  # Stereo audio
        video_sf = video_sf.mean(axis=1)  ## <-- converts to mono
        print("Converted stereo to mono")
    
    audio_duration = len(video_sf) / sfsr
    video_stream = container.streams.video[0]

    audio_window = 1/10  ## try increasing audio_window (optional)
    frame_window = 1/30
    next_t = 0.0
    frame_count = 0
    expected_audio_length = int(audio_window * sfsr)
    
    print(f"Expected audio segment length: {expected_audio_length} samples")
    
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
        if not result.face_landmarks or len(result.face_landmarks) == 0:
            print(f"WARNING: no face detected at frame {frame_count}, time {t}s - using zeros")
            face_features = np.zeros(1434)  ## 478(?) landmarks * 3 coords
        else:
            face = result.face_landmarks[0]
            face_features = np.array([[lm.x, lm.y, lm.z] for lm in face]).flatten()

        if frame_count % 3 == 0 and t + audio_window <= audio_duration:
            start_sample = int(t * sfsr)
            end_sample = int((t + audio_window) * sfsr)
            
            ## bound checking (ti)
            if end_sample <= len(video_sf):
                segment = video_sf[start_sample:end_sample]
                
                print(f"Frame {frame_count}: segment length = {len(segment)} samples at time {t}s")  # DEBUG
                
                if len(segment) >= expected_audio_length * 0.8:  ## needs 80% of len
                    if len(segment) < expected_audio_length:
                        segment = np.pad(segment, (0, expected_audio_length - len(segment)), mode='constant')
                    current_audio_segment = segment
                else:
                    print(f"WARNING: segment too short ({len(segment)} samples), skipping audio update")

        voice_features = smile.process_signal(
            current_audio_segment,
            sfsr
        ).to_numpy().flatten()

        next_t += frame_window
        frame_count += 1

        time_step = np.concatenate((face_features, voice_features))
        print(f"Frame {frame_count}: face_features shape={face_features.shape}, voice_features shape={voice_features.shape}, total shape={time_step.shape}")
        time_series.append(time_step)
        
        ## progress tracking
        if frame_count % 100 == 0:
            print(f"Ran {frame_count} frames...")

    print(f"\nTotal frames ran: {frame_count}")
    return np.array(time_series)

if __name__ == "__main__":
    print(mead_feature_list("/home/rashadwsl/projects/mead-repo/micro-expression-analysis-device-mead/micro_expression_analysis_device_mead/utterance_1.mp4"))
