import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import h5py
import os
import numpy as np
import pandas as pd
import json
import time
import sqlite3

from mne_reader import df, create_db_connection, read_eeg, get_eeg_data, preprocess_eeg, extract_eeg_features, eeg_to_image, plot_eeg_img

def create_connection(db_file_name):
    conn = None

    try:
        conn = sqlite3.connect(db_file_name)
    except sqlite3.Error as e:
        print(e)

    return conn

df = df[df["recording_duration"] > 10]

h5f = h5py.File(f"/mnt/disks/data/files/TUSZ_files/raw_data_TUSZ.h5", 'w')

conn = create_connection("/mnt/disks/data/files/TUSZ_files/eeg_recordings_TUSZ.db")

with open('data/JSON_files/TUSZ/TUSZ_annotations.json', 'r') as f:
    annotations = json.load(f)


for index, data in df.iterrows():
    eeg_file_name = data["file_name"]
    eeg_file_path = data["file_path"]

    print(f"({index}/{df.shape[0]}) {eeg_file_name}")

    #print("   ", "Processing...")
    eeg = read_eeg(eeg_file_name, eeg_file_path='/mnt/disks/data/files/TUSZ_files' + eeg_file_path)
    eeg = eeg.resample(256)
    # Set lowpass value to None to allow plotting of signal
    eeg.info['lowpass'] = None


    #
    #   PREPROCESS EEG CHANNEL WITH FILTERS
    #
    eeg_prep = preprocess_eeg(eeg)
    eeg_prep_data, _ = get_eeg_data(eeg_prep)

    plot_eeg(eeg_prep)

h5f.close()
