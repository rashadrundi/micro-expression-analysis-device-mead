from __future__ import print_function
import argparse
import os
import json
import glob
import sys
import subprocess
import datetime
import numpy as np
from ..mead_feature_extract_main import mead_feature_list


def get_formatted_time(seconds):
    return str(datetime.timedelta(seconds=seconds))

#    microsecond = int((seconds - int(seconds)) * 1000 * 1000)
#    int_seconds = int(seconds)
#    hour = int_seconds // 3600
#    minute = (int_seconds - hour * 3600) // 60
#    second = int_seconds - hour * 3600 - minute * 60
#    return "{:02}:{:02}:{:02}.{:03}".format(hour, minute, second, microsecond)

def youtube_available(url):
    try:
        result = subprocess.run(
            ["yt-dlp", "-j", url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return result.returncode == 0
    except FileNotFoundError:
        raise RuntimeError("yt-dlp not installed!")

def dl_youtube(link, target_file):
    p = subprocess.Popen(["yt-dlp",
                          "-f", "best",
                          "--merge-output-format", "mp4",
                          "--restrict-filenames",
                          "--socket-timeout", "20",
                          "-iwc",
                          "--write-info-json",
                          '--write-annotations',
                          '--prefer-ffmpeg',
                          link,
                          '-o', target_file],
                         )
    out, err = p.communicate()

def prepare_data(file, target_dir, json_savefile, feature_save_dir):
    meta = []

    temp_directory = os.path.abspath(os.path.join(target_dir, "youtube_videos_temp"))
    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)

    print ("here")
    increment = 0
    with open(file) as f:
        next(f)
        for l in f:
            l = l.strip()
            if len(l) > 0:
                link, start, end, video, utterance, arousal, valence = l.split(',')[:7]

                if not youtube_available(link):
                    print(f"[SKIP] Cannot use video: {link}")
                    continue 

		#print "Link:", link

                result_dir = os.path.join(os.path.join(target_dir, video))
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                result_filename = os.path.abspath(os.path.join(result_dir, utterance))
                #dl video with youtube-dl

                target_file = os.path.abspath(os.path.join(temp_directory, video + ".mp4"))
                if not os.path.exists(target_file):
                    dl_youtube(link, target_file)

                duration = float(end) - float(start)  ## get duration

                p = subprocess.call(["ffmpeg",
                    "-y",
                    "-ss", get_formatted_time(float(start)),  ## seek BEFORE input
                    "-i", target_file,
                    "-t", get_formatted_time(duration),        ## we need duration before
                    "-c:a", "aac",
                    '-strict', '-2',
                    result_filename],
                    )
                
                feature_list = mead_feature_list(result_filename)
                np.save(os.path.join(feature_save_dir, f"{video}_{utterance}.npy"), feature_list)

                meta.append({
                    "feature_path": os.path.join(feature_save_dir, f"{video}_{utterance}.npy"),
                    "v" : valence,
                    "a": arousal
                })

                os.remove(result_filename)
                increment += 1
                print(f"------------------------------------------ \n \n Row {increment} done \n \n -------------------------------------------")
                if increment % 10 == 0:
                    json.dump(meta, open(json_savefile, "w"), indent=2)
                if increment == 400:
                    sys.exit()



if __name__ == "__main__":
    THIS_DIR = os.path.dirname(__file__)

    csv_file = os.path.join(THIS_DIR, "omg_TrainVideos.csv")
    temp_dir = os.path.join(THIS_DIR, "temp_videos")
    metadata_file = os.path.join(THIS_DIR, "train_metadata.json")
    features_out = os.path.join(THIS_DIR, "features_train")

    prepare_data(csv_file, temp_dir, metadata_file, features_out)
    prepare_data("omg_TrainVideos.csv", "temp_videos", "train_metadata.json", "feature_train/")





                
