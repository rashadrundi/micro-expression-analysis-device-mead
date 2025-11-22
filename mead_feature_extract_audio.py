import opensmile

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

extracted_features = smile.process_file("test_audio/audio1.wav").to_numpy()

print(extracted_features)
