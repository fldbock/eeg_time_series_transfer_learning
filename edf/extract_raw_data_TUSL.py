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

h5f = h5py.File(f"/mnt/disks/data/files/TUSL_files/raw_data_TUSL.h5", 'w')

conn = create_connection("/mnt/disks/data/files/TUSL_files/eeg_recordings_TUSL.db")

with open('data/JSON_files/TUSL/TUSL_annotations.json', 'r') as f:
    annotations = json.load(f)

print("Start")

for index, data in df.iterrows():
    eeg_file_name = data["file_name"]
    eeg_file_path = data["file_path"]

    print(f"({index}/{df.shape[0]}) {eeg_file_name}")

    #print("   ", "Processing...")
    eeg = read_eeg(eeg_file_name, eeg_file_path='/mnt/disks/data/files/TUSL_files' + eeg_file_path)
    eeg = eeg.resample(256)
    # Set lowpass value to None to allow plotting of signal
    eeg.info['lowpass'] = None


    #
    #   PREPROCESS EEG CHANNEL WITH FILTERS
    #
    eeg_prep = preprocess_eeg(eeg)
    hz = 256
    eeg = eeg.resample(hz)#delete later
    eeg_prep_data, _ = get_eeg_data(eeg_prep)

    #
    #   SPLIT RAW DATA IN 10s INTERVALS
    #
    
    if eeg_file_name[:-4] in annotations:
        windows_list = ""
        ##split where annotatted for loop
        splits = annotations[eeg_file_name[:-4]]
        i = -1
        for split in splits:
            split = split.split(" ")
            start_time = float(split[0])
            stop_time = float(split[1])
            label = split[2]

            eeg_prep_windows_data = []
            window_width = 10
            windows = np.arange(0, data["recording_duration"], window_width/4)#/4

            number_of_windows = 0
            for index in windows:
                if int((index + window_width + start_time) * hz) <= int(stop_time * hz):
                    #print("         ", "Window is", index * 256, "to", (index + window_width) * 256)
                    eeg_prep_window_data = eeg_prep_data[:, int((start_time + index) * hz):int((index + window_width+ start_time) * hz)]
                    eeg_prep_windows_data.append(eeg_prep_window_data)
                    
                    ##add one to window counter
                    number_of_windows += 1
            i += 1
            
            windows_list = windows_list + " " + str(number_of_windows)
            #
            #   SAVE TO H5Py
            #
            print(eeg_file_name[:-4])
            h5f.create_dataset(eeg_file_name+"_"+str(i), data=eeg_prep_windows_data)
        
        sql = """UPDATE tokens SET number_of_windows = ? WHERE token_id = ?"""
        cur = conn.cursor()
        cur.execute(sql, (str(windows_list), eeg_file_name[:-4]))
        cur.close()
        conn.commit()

h5f.close()
