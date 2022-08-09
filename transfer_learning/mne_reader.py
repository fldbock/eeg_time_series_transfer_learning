import math

import matplotlib.pyplot as plt
from matplotlib import animation
#%matplotlib inline
#%matplotlib notebook

import mne
import mne.filter

import numpy as np

import pandas as pd

import pywt

import scipy
from scipy import stats
from scipy import signal
from scipy.integrate import simps
from scipy.signal import welch

import sklearn
from sklearn import preprocessing

import sqlite3
import os
from pathlib import Path


#
#   CONNECT TO LOCAL DATABASE
#
def create_db_connection(db_file_name):
    conn = None

    try:
        conn = sqlite3.connect(db_file_name)
    except sqlite3.Error as e:
        print(e)

    return conn



conn = create_db_connection('/mnt/disks/data/files/TUSZ_files/eeg_recordings_TUSZ.db')

df = pd.DataFrame(pd.read_sql_query("SELECT patients.patient_id, sessions.session_id, tokens.token_id, tokens.diagnosis, sessions.electrode_setup, tokens.sampling_freq, tokens.nr_of_samples, tokens.len_of_samples, tokens.recording_duration, tokens.eeg_chs, tokens.non_eeg_chs, tokens.file_path, tokens.file_name FROM patients INNER JOIN sessions ON patients.patient_id == sessions.patient_id INNER JOIN tokens ON sessions.patient_id == tokens.patient_id AND sessions.session_id == tokens.session_id", conn))

#
#   DEFINE USEFUL FUNCTIONS
#
# Apply standard 10-20 montage to EEG recording to allow azimuth equidistant projection
def apply_eeg_montage(eeg, montage):
    def standardize_eeg_ch_name(ch_name): # Removes "EEG" prefices and "-Ref" suffices in EEG channel names
        ch_name = ch_name[4:ch_name.find("-")]
        ch_name_old = ch_name

        if ch_name[0] != 'Z':
            ch_name = ch_name.replace("Z", "z")

        if ch_name[0] != 'P':
            ch_name = ch_name.replace("P", "p")

        return ch_name

    # Rename channel names of recording to fit montage
    eeg_montage = eeg.copy()
    eeg_montage.rename_channels(standardize_eeg_ch_name)

    # Apply montage to recording
    eeg_montage.set_montage(montage)

    return eeg_montage


# Read in EEG from EDF file
def read_eeg(eeg_file_name, eeg_file_path=""):
    # Define EEG channels to keep
    eeg_chs_in_montage = ['EEG C3', 'EEG C4', 'EEG CZ', 'EEG F3', 'EEG F4', 'EEG F7', 'EEG F8', 'EEG FP1', 'EEG FP2', 'EEG FZ', 'EEG O1', 'EEG O2', 'EEG P3', 'EEG P4', 'EEG PZ', 'EEG T3', 'EEG T4', 'EEG T5', 'EEG T6']

    # Read in non-EEG channels
    non_eeg_chs_str = df.loc[df['file_name'] == eeg_file_name, 'non_eeg_chs'].iloc[0]
    non_eeg_chs = non_eeg_chs_str[1:-1].split(',')
    non_eeg_chs = [non_eeg_ch.strip()[1:-1] for non_eeg_ch in non_eeg_chs]

    # Read in EEG channels
    eeg_chs_str = df.loc[df['file_name'] == eeg_file_name, 'eeg_chs'].iloc[0]
    eeg_chs = eeg_chs_str[1:-1].split(',')
    eeg_chs = [eeg_ch.strip()[1:-1] for eeg_ch in eeg_chs]

    # Define EEG channels to exclude
    eeg_chs_not_in_montage = [eeg_ch for eeg_ch in eeg_chs if eeg_ch[:eeg_ch.index("-")] not in eeg_chs_in_montage]
    eeg_chs_to_exclude = list(set(eeg_chs_not_in_montage) | set(non_eeg_chs))
    if eeg_file_name[-4:] != ".edf":
        if Path(eeg_file_path + eeg_file_name).exists():
            os.rename(eeg_file_path + eeg_file_name,eeg_file_path + eeg_file_name + '.edf')
        eeg = mne.io.read_raw_edf(eeg_file_path + eeg_file_name + '.edf', exclude=eeg_chs_to_exclude, preload=True, verbose=False)
    else:
        eeg = mne.io.read_raw_edf(eeg_file_path + eeg_file_name, exclude=eeg_chs_to_exclude, preload=True, verbose=False)

    montage_1020 = mne.channels.make_standard_montage('standard_1020')
    eeg_montage = apply_eeg_montage(eeg, montage_1020)

    return eeg_montage


# Return important EEG metadata
def get_eeg_metadata(eeg):
    # Number of channels in recording
    eeg_nr_of_chs = len(eeg.info["chs"])

    # Number of samples/timestamps in recording
    eeg_nr_of_timestamps = eeg.n_times

    # Sample frequency of recording
    eeg_sampling_freq = eeg.info["sfreq"]


    return eeg_nr_of_chs, eeg_nr_of_timestamps, eeg_sampling_freq


# Return EEG data and timestamps
def get_eeg_data(eeg):
    # TO DO: Change reference of EEG channels to one of POL channels?
    eeg_data, eeg_timestamps = eeg.get_data(return_times=True)
    return eeg_data, eeg_timestamps


# Return EEG data and timestamps for particular channels
def get_eeg_chs_data(eeg, ch_names):
    eeg_data, eeg_timestamps = eeg.get_data(picks=ch_names, return_times=True)
    return eeg_data, eeg_timestamps


# Plot the n first channels of the EEG recording
def plot_eeg_n_chs(eeg, n_chs):
    eeg.plot(start=125, duration=3, scalings=dict(eeg=21e-4), n_channels=n_chs, remove_dc=True)


# Plot a single channel of the EEG recording
def plot_eeg_ch(eeg, ch_name):
    eeg_ch_names = eeg.ch_names

    eeg_ch_idx = eeg_ch_names.index(ch_name)
    if eeg_ch_idx < 0:
        print(f"Channel name must be in {eeg_ch_names}.")
        return

    eeg_data, eeg_timestamps = get_eeg_data(eeg)
    eeg_nr_of_chs, eeg_nr_of_timestamps, eeg_sampling_freq = get_eeg_metadata(eeg)

    eeg_ch_data = eeg_data[eeg_ch_idx]
    window_start, window_end = 24.5, 25

    plt.plot(eeg_timestamps, (eeg_ch_data))
    plt.xlim(window_start, window_end)
    plt.ylim(-0.002, 0.002)
    # Plot using https://www.cs.colostate.edu/eeg/data/json/doc/tutorial/_build/html/getting_started.html

    return


# Returns all EEG electrode names and locations
def get_eeg_electrode_coords(eeg):
    montage_1020 = mne.channels.make_standard_montage('standard_1020')

    electrode_coords = {}
    electrodes = eeg.info["dig"]

    for electrode in electrodes:
        if str(electrode["kind"]) == "3 (FIFFV_POINT_EEG)":
            # Read in electrode identifier
            electrode_nr = electrode["ident"]

            # Map electrode identifier to corresponding channel name
            electrode_ch_name = montage_1020.ch_names[electrode_nr-1]

            # Map electrode channel name to 3D electrode locations
            electrode_coords[electrode_ch_name] = electrode["r"][0:3]

    return electrode_coords


# Map 3D electrode locations to 2D locations via Azimuth Equidistant Projection
def get_eeg_electrodes_projs(eeg):
    # https://github.com/pprakhar30/EEGSignalAnalysis/blob/master/EEG_Model.py
    def cart2sph(x, y, z):
        x2_y2 = x**2 + y**2
        r = np.sqrt(x2_y2 + z**2)                    # r
        elev = np.arctan2(z, np.sqrt(x2_y2))            # Elevation
        az = np.arctan2(y, x)                          # Azimuth
        return r, elev, az

    def pol2cart(theta, rho):
        return rho * np.cos(theta), rho * np.sin(theta)

    # Project electrode locations to preserve distance
    electrode_projs = {}
    electrode_coords = get_eeg_electrode_coords(eeg)

    for ch_name in eeg.ch_names:
        # Retrieve original carthesian electrode coordinates
        x, y, z = electrode_coords[ch_name]

        # Project carthe
        [r, elev, az] = cart2sph(x, y, z)
        x_proj, y_proj = pol2cart(az, np.pi / 2 - elev)

        electrode_projs[ch_name] = [x_proj, y_proj]

    return electrode_projs


# Plot Azimuth Equidistant Projection of electrode locations
def plot_eeg_electrode_projs(eeg, electrode_projs):
    electrode_projs = get_eeg_electrodes_projs(eeg)
    electrode_projs = shift_eeg_electrode_projs(electrode_projs, 'Cz')

    x_proj = [electrode_projs[ch_name][0] for ch_name in electrode_projs.keys()]
    y_proj = [electrode_projs[ch_name][1] for ch_name in electrode_projs.keys()]

    fig, ax = plt.subplots()
    ax.scatter(x_proj, y_proj)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.gca().set_aspect("equal", adjustable="box")
    for i, ch_name in enumerate(electrode_projs.keys()):
        ax.annotate(ch_name, (x_proj[i], y_proj[i]))


# Shift projected electrode coordinates to center them on a specific electrode, e.g., EEG Cz-Ref (Cz for short)
def shift_eeg_electrode_projs(electrode_projs, ref_electrode):
    x_shift, y_shift = electrode_projs[ref_electrode]

    shifted_electrode_projs = electrode_projs.copy()
    for electrode_proj in electrode_projs.keys():
        x, y = electrode_projs[electrode_proj]
        shifted_electrode_projs[electrode_proj] =  [x - x_shift, y - y_shift]

    return shifted_electrode_projs


# Get value of EEG sample
def get_eeg_value(eeg, timestamp):
    eeg_data, _ = get_eeg_data(eeg)
    return [(value * 1e6) for value in list(eeg_data[:, min(timestamp, eeg.n_times)])] # Convert V to µV


# Normalize electrode positions to range(0, res-1)
def normalize_electrode_projs(electrode_projs_points, res):
    x_min, y_min = np.min(electrode_projs_points, 0)
    x_max, y_max = np.max(electrode_projs_points, 0)

    electrode_projs_points = [[(res - 1) * (x - x_min) / (x_max - x_min), (res - 1) * (y - y_min) / (y_max - y_min)] for x, y in electrode_projs_points]

    return electrode_projs_points


# Construct image from electrode positions and EEG values
def eeg_to_image(eeg, eeg_values, res, interpolate=True):
    # Define translator
    eeg_ch_to_idx = {}
    for ch_idx, ch in enumerate(eeg.ch_names):
        eeg_ch_to_idx[ch] = ch_idx

    # Read in electrode positions
    electrode_projs = get_eeg_electrodes_projs(eeg)
    electrode_projs_points = [electrode_projs[ch_name] for ch_name in electrode_projs.keys()]

        # Read in values
    electrode_projs_values = eeg_values

    # Normalize electrode values
    electrode_projs_values = list(electrode_projs_values)


    # Normalize electrode positions
    electrode_projs_points = normalize_electrode_projs(electrode_projs_points, res)

    # Add values to edges of image
    if interpolate:
        electrode_projs_points.extend([
            [0, 0],
            [0, res-1],
            [res-1, 0],
            [res-1, res-1],
            [res/2, 0],
            [res/2, res-1],
            [0, res/2],
            [res-1, 0]
        ])

        electrode_projs_values.extend([
            electrode_projs_values[eeg_ch_to_idx["T5"]],
            electrode_projs_values[eeg_ch_to_idx["F7"]],
            electrode_projs_values[eeg_ch_to_idx["T6"]],
            electrode_projs_values[eeg_ch_to_idx["F8"]],
            (electrode_projs_values[eeg_ch_to_idx["O1"]] + electrode_projs_values[eeg_ch_to_idx["O2"]]) / 2,
            (electrode_projs_values[eeg_ch_to_idx["Fp1"]] + electrode_projs_values[eeg_ch_to_idx["Fp2"]]) / 2,
            electrode_projs_values[eeg_ch_to_idx["T3"]],
            electrode_projs_values[eeg_ch_to_idx["T4"]],
        ])

    # Interpolate values of electrode projections and create interpolator, i.e., object that generates intermediate values
    interpolation_tol = 1e-6
    interpolator = scipy.interpolate.CloughTocher2DInterpolator(electrode_projs_points, electrode_projs_values, tol=interpolation_tol)

    # Sample interpolation space to form image
    img = np.zeros((res, res))

    for i_idx, i in enumerate(range(res)):
        for j_idx, j in enumerate(range(res)):
            img_pixel_value = interpolator([i, j])[0]
            if np.isnan(img_pixel_value):
                #img_pixel_value = 0
                pass

            img[i_idx, j_idx] = img_pixel_value
    return img


# Show EEG image
def plot_eeg_img(eeg, img, title=None, normalize=False):
    fig, ax = plt.subplots()

    if normalize:
        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

    im = ax.imshow(img.T, cmap="viridis", vmin=0, vmax=1)
    plt.gca().set_aspect("equal", adjustable="box")

    # Overlay labels over EEG image
    electrode_projs = get_eeg_electrodes_projs(eeg)
    electrode_projs_points = [electrode_projs[ch_name] for ch_name in electrode_projs.keys()]
    electrode_projs_points = normalize_electrode_projs(electrode_projs_points, img.shape[0])

    for i, ch_name in enumerate(electrode_projs.keys()):
        ax.annotate(ch_name, (electrode_projs_points[i][0], electrode_projs_points[i][1]))

    plt.xlim(-0.5, img.shape[0]-0.5)
    plt.ylim(-0.5, img.shape[1]-0.5)

    if title:
        plt.title(title)

    plt.show()


# Preprocess EEG data
def preprocess_eeg(eeg):
    eeg_filtered = eeg.copy()

    # Slight noise bump at 60 Hz but not substantial
    eeg_filtered.notch_filter(np.arange(60,eeg.info['sfreq']/2, 60),filter_length='auto', phase='zero', verbose=False)
    eeg_filtered.filter(l_freq=0, h_freq=min(128, int(eeg_filtered.info["sfreq"]/2 - 1)), verbose=False)
    eeg_filtered.filter(l_freq=0.5, h_freq=None, verbose=False)

    # TO DO: Remove artifacts with ICA
    """
    ica = mne.preprocessing.ICA(n_components=2, random_state=97, max_iter=800, verbose=0)
    ica.fit(eeg_filtered)
    eeg_prep = eeg_filtered.copy()
    ica.apply(eeg_prep, exclude=[1])
    """
    eeg_prep = eeg_filtered

    return eeg_prep


# PyREM (https://github.com/gilestrolab/pyrem/blob/master/src/pyrem/univariate.py)
def extract_hjorth_features(a):
    first_deriv = np.diff(a)
    second_deriv = np.diff(a,2)

    var_zero = np.mean(a ** 2)
    var_d1 = np.mean(first_deriv ** 2)
    var_d2 = np.mean(second_deriv ** 2)

    activity = var_zero
    morbidity = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / morbidity

    return activity, morbidity, complexity


def extract_eeg_features(eeg, band_power_coarse=True, band_power_fine=True, hjorth=True, stat_moments=True, wavelet=True):
    # https://ieeexplore.ieee.org/abstract/document/8848612

    # BAND POWER FEATURES
    # https://raphaelvallat.com/bandpower.html
    band_feature_names = []
    band_features = []

    freq_bands = []
    if band_power_coarse:
        freq_bands.extend([
            ["delta", 1, 4],
            ["theta", 4, 8],
            ["alpha", 8, 13],
            ["beta", 13, 30],
            ["gamma-1", 30, min(128, int(eeg.info["sfreq"]/2 - 1))]
        ])

    if band_power_fine:
        min_freq = 0.5
        max_freq = 50
        band_width = 0.5

        for band_min in np.arange(min_freq, max_freq + band_width, band_width):
            band_max = band_min + band_width

            if band_min == max_freq:
                band_max = min(128, int(eeg.info["sfreq"]/2 - 1))

            band_name = f"{band_min}-{band_max}"
            freq_bands.append([band_name, band_min, band_max])

    # Calculate total power of signal to relate (relative) band power to
    freqs, psd = signal.welch(eeg.get_data(), eeg.info["sfreq"], nperseg=(eeg.n_times))
    freq_res = freqs[1] - freqs[0]
    total_power = simps(psd, dx=freq_res)

    for band, band_min, band_max in freq_bands:
        # Plot channel signals in frequency band
        eeg_band = eeg.copy()
        eeg_band = eeg_band.filter(l_freq=band_min, h_freq=band_max, verbose=False)

        # Calculate power spectral density of current band per channel
        band_freqs, band_psd = signal.welch(eeg_band.get_data(), eeg_band.info["sfreq"], nperseg=(eeg_band.n_times))

        # Calculate average band power per channel with Simpson
        freq_res = band_freqs[1] - band_freqs[0]
        band_power = simps(band_psd, dx=freq_res)
        rel_band_power = band_power / total_power

        # Add feature to list
        band_feature_names.append(f"{band}_band_power")
        band_features.append(rel_band_power)

    # HJORTH FEATURES
    hjorth_feature_names = []
    hjorth_features = []
    if hjorth:
        activities, complexities, morbidities = [], [], []

        for ch in eeg.ch_names:
            eeg_ch_data = eeg.get_data(picks=[ch])
            activity, complexity, morbidity = extract_hjorth_features(eeg_ch_data)
            activities.append(activity)
            complexities.append(complexity)
            morbidities.append(morbidity)

        hjorth_feature_names.extend(("hjorth activity", "hjorth complexity", "hjorth morbidity"))
        hjorth_features.extend((activities, complexities, morbidities))

    # STATISTICAL MOMENT FEATURES
    stats_feature_names = []
    stats_features = []
    if stat_moments:
        eeg_data = eeg.get_data()

        # Mean
        mean = np.mean(eeg_data, axis=1)
        stats_feature_names.append("mean")
        stats_features.append(mean)

        # Standard deviation
        std = np.std(eeg_data, axis=1)
        stats_feature_names.append("standard deviation")
        stats_features.append(std)

        # Variance
        variance = np.var(eeg_data, axis=1)
        stats_feature_names.append("variance")
        stats_features.append(variance)

        # Skewness
        skewness = stats.skew(eeg_data, axis=1)
        stats_feature_names.append("skewness")
        stats_features.append(skewness)

        # Kurtosis
        kurtosis = stats.kurtosis(eeg_data, axis=1)
        stats_feature_names.append("kurtosis")
        stats_features.append(kurtosis)

        # Energy
        energy = np.sum(abs(eeg_data)**2, axis=1)
        stats_feature_names.append("energy")
        stats_features.append(energy)

        # Max
        max_ = np.amax(eeg_data, axis=1)
        stats_feature_names.append("max")
        stats_features.append(max_)


    # WAVELET FEATURES
    wavelet_feature_names = []
    wavelet_features = []

    if wavelet:
        eeg_data = eeg.get_data()

        # Each channel has DWT; DWT's are longer for shallower levels
        # https://pywavelets.readthedocs.io/en/0.2.2/regression/multilevel.html
        db4 = pywt.Wavelet('db4')
        cA5, cD5, cD4, cD3, cD2, cD1 = pywt.wavedec(eeg_data, db4, level=5)

        # Find envelope for wavelet subbands via Hilbert transform
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
        sA5_temp = scipy.signal.hilbert(cA5)
        sA5 = np.abs(sA5_temp)

        sD5_temp = scipy.signal.hilbert(cD5)
        sD5 = np.abs(sD5_temp)

        sD4_temp = scipy.signal.hilbert(cD4)
        sD4 = np.abs(sD4_temp)

        sD3_temp = scipy.signal.hilbert(cD3)
        sD3 = np.abs(sD3_temp)

        sD2_temp = scipy.signal.hilbert(cD2)
        sD2 = np.abs(sD2_temp)

        sD1_temp = scipy.signal.hilbert(cD1)
        sD1 = np.abs(sD1_temp)

        wavelets_data = [sA5, sD5, sD4, sD3, sD2, sD1]
        wavelets_names= ["sA5", "sD5", "sD4", "sD3", "sD2", "sD1"]

        for wavelet_idx in range(len(wavelets_data)):
            wavelet_data = wavelets_data[wavelet_idx]
            wavelet_name = wavelets_names[wavelet_idx]

            # Mean
            wavelet_mean = np.mean(wavelet_data, axis=1)
            wavelet_feature_names.append(f"{wavelet_name} wavelet mean")
            wavelet_features.append(wavelet_mean)

            # Standard deviation
            wavelet_std = np.std(wavelet_data, axis=1)
            wavelet_feature_names.append(f"{wavelet_name} wavelet standard deviation")
            wavelet_features.append(wavelet_std)

            # Energy
            wavelet_energy = np.sum(abs(wavelet_data)**2, axis=1)
            wavelet_feature_names.append(f"{wavelet_name} wavelet energy")
            wavelet_features.append(wavelet_energy)

            # Max
            wavelet_max = np.amax(wavelet_data, axis=1)
            wavelet_feature_names.append(f"{wavelet_name} wavelet max")
            wavelet_features.append(wavelet_max)

    # Add features
    feature_names = []
    feature_names.extend(band_feature_names)
    feature_names.extend(hjorth_feature_names)
    feature_names.extend(stats_feature_names)
    feature_names.extend(wavelet_feature_names)

    features = []
    features.extend(band_features)
    features.extend(hjorth_features)
    features.extend(stats_features)
    features.extend(wavelet_features)

    # Plot features
    # for feature_idx in range(len(features)):
    #     feature_img = eeg_to_image(eeg, features[feature_idx], 16)
    #     plot_eeg_img(eeg, feature_img, feature_names[feature_idx], normalize=True)

    return feature_names, features