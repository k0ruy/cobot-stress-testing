import os
import pandas as pd
from tqdm import tqdm
import neurokit2 as nk
from neurokit2.signal import signal_smooth, signal_zerocrossings
from pyEDA.main import *
from scipy.signal import find_peaks
from scipy.fftpack import fft, ifft, fftfreq
import pywt
import scipy
from librosa import feature
import pyhrv
import warnings

# suppress warnings
warnings.filterwarnings("ignore")


def import_opensignals(filename):
    with open(filename, 'r') as sig_file:
        header = sig_file.readline()
        header = sig_file.readline()
    sig_file.close()

    # Find signals present in the log
    hparts = header.split("sensor\": [\"")
    hpart = hparts[1].split("\"], \"label")
    signals_ids = hpart[0].split("\", \"")
    nsig = len(signals_ids)
    signals_ids_all = signals_ids.copy()
    # Manage known case where only the second instance of ECG shall be retained
    if nsig > len(np.unique(signals_ids)):
        dup = [i for i in np.arange(nsig) if signals_ids.count(signals_ids[i]) > 1]
        signals_ids_all[dup[0]] = signals_ids_all[dup[0]] + '1'
        signals_ids.pop(dup[0])
    sig_names = ['time', 'none'] + signals_ids_all
    sig_cols = ['time'] + signals_ids

    signals = pd.read_csv(filename, sep='[\t]', header=None, index_col=False,
                          names=sig_names, skiprows=3, usecols=sig_cols, engine='python')

    # Find framerate
    hparts = header.split("sampling rate\": ")
    hpart = hparts[1].split(", \"resolution")
    fs = int(hpart[0])

    return signals, fs


def makedirs(path):
    """
    create directory on the "path name" """

    if not os.path.exists(path):
        os.makedirs(path)


def import_filenames(directory_path):
    """
    import all file names of a directory """
    filename_list = []
    dir_list = []
    for root, dirs, files in os.walk(directory_path, topdown=False):
        filename_list = files
        dir_list = dirs
    return filename_list, dir_list


def normalize(x, x_mean, x_std):
    """
    perform z-score normalization of a signal """
    x_scaled = (x - x_mean) / x_std
    return x_scaled


def make_window(signal, fs, overlap, window_size_sec):
    """
    perform cropped signals of window_size seconds for the whole signal
    overlap input is in percentage of window_size
    window_size is in seconds """

    window_size = fs * window_size_sec
    overlap = int(window_size * (overlap / 100))
    start = 0
    segmented = np.zeros((1, window_size), dtype=int)
    while (start + window_size <= len(signal)):
        segment = signal[start:start + window_size]
        segment = segment.reshape(1, len(segment))
        segmented = np.append(segmented, segment, axis=0)
        start = start + window_size - overlap
    return segmented[1:]


def extract_time_and_freq_hrv_features(dataframe, fs):
    features_df = pd.DataFrame(columns=['HR_Mean',
                                        'QRS_Mean',
                                        'QR_Mean',
                                        'RS_Mean',
                                        'PR_Mean',
                                        'TR_Mean',
                                        'PP_Mean',
                                        'QQ_Mean',
                                        'SS_Mean',
                                        'TT_Mean',
                                        'HRV_MeanNN',
                                        'HRV_SDNN',
                                        'HRV_SDANN1',
                                        'HRV_SDNNI1',
                                        'HRV_SDANN2',
                                        'HRV_SDNNI2',
                                        'HRV_SDANN5',
                                        'HRV_SDNNI5',
                                        'HRV_RMSSD',
                                        'HRV_SDSD',
                                        'HRV_CVNN',
                                        'HRV_CVSD',
                                        'HRV_MedianNN',
                                        'HRV_MadNN',
                                        'HRV_MCVNN',
                                        'HRV_IQRNN',
                                        'HRV_Prc20NN',
                                        'HRV_Prc80NN',
                                        'HRV_pNN50',
                                        'HRV_pNN20',
                                        'HRV_MinNN',
                                        'HRV_MaxNN',
                                        'HRV_HTI',
                                        'HRV_TINN',
                                        'VLF_Peak',
                                        'VLF_Rel',
                                        'LF_Peak',
                                        'LF_Rel',
                                        'HF_Peak',
                                        'HF_Rel',
                                        'LF_Norm',
                                        'HF_Norm',
                                        'fft_ratio',
                                        'fft_total'])

    for i in tqdm(range(0, dataframe.shape[0])):
        # processed_ecg, _ = nk.process(dataframe[i,3:], sampling_rate=fs)
        cleaned_ecg = nk.ecg_clean(dataframe[i, 1:], sampling_rate=fs, method='biosppy')
        R_peaks, R_info = nk.ecg_peaks(cleaned_ecg, sampling_rate=fs)

        hrv_time = nk.hrv_time(R_peaks, sampling_rate=fs, show=False)
        hrv_freq, _, _ = pyhrv.frequency_domain.welch_psd(R_info["ECG_R_Peaks"] / fs, show=False, mode='dev')
        hrv_freq = hrv_freq.as_dict()
        hr_mean = np.array(nk.ecg_rate(R_peaks, sampling_rate=fs,
                                       desired_length=len(R_peaks)).mean())

        pqst_peaks, pqst_info = nk.ecg_delineate(cleaned_ecg, R_info, sampling_rate=fs)
        pqst_bool = pqst_peaks.to_numpy()

        R_peak_time = R_info["ECG_R_Peaks"]
        P_peak_time = pqst_info["ECG_P_Peaks"]
        Q_peak_time = pqst_info["ECG_Q_Peaks"]
        S_peak_time = pqst_info["ECG_S_Peaks"]
        T_peak_time = pqst_info["ECG_T_Peaks"]

        R_peak_time = np.array(R_peak_time) * (1 / fs)
        P_peak_time = np.array(P_peak_time) * (1 / fs)
        Q_peak_time = np.array(Q_peak_time) * (1 / fs)
        S_peak_time = np.array(S_peak_time) * (1 / fs)
        T_peak_time = np.array(T_peak_time) * (1 / fs)

        QRS_duration = S_peak_time - Q_peak_time
        QR_duration = R_peak_time - Q_peak_time
        RS_duration = S_peak_time - Q_peak_time
        PR_duration = R_peak_time - P_peak_time
        TR_duration = T_peak_time - R_peak_time

        delta_PP_time = np.diff(P_peak_time)
        delta_QQ_time = np.diff(Q_peak_time)
        delta_SS_time = np.diff(S_peak_time)
        delta_TT_time = np.diff(T_peak_time)

        mean_QRS_duration = np.nanmean(QRS_duration)
        mean_QR_duration = np.nanmean(QR_duration)
        mean_RS_duration = np.nanmean(RS_duration)
        mean_PR_duration = np.nanmean(PR_duration)
        mean_TR_duration = np.nanmean(TR_duration)

        mean_PP_time = np.nanmean(delta_PP_time)
        mean_QQ_time = np.nanmean(delta_QQ_time)
        mean_SS_time = np.nanmean(delta_SS_time)
        mean_TT_time = np.nanmean(delta_TT_time)

        temp = np.hstack((hr_mean,
                          mean_QRS_duration,
                          mean_QR_duration,
                          mean_RS_duration,
                          mean_PR_duration,
                          mean_TR_duration,
                          mean_PP_time,
                          mean_QQ_time,
                          mean_SS_time,
                          mean_TT_time,
                          hrv_time.loc[0],
                          hrv_freq["fft_peak"][0],
                          hrv_freq["fft_rel"][0],
                          hrv_freq["fft_peak"][1],
                          hrv_freq["fft_rel"][1],
                          hrv_freq["fft_peak"][2],
                          hrv_freq["fft_rel"][2],
                          hrv_freq["fft_norm"][0],
                          hrv_freq["fft_norm"][1],
                          hrv_freq["fft_ratio"],
                          hrv_freq["fft_total"]))
        features_df.loc[i] = temp

    return features_df


def extract_eda_time_and_frequency_features(dataframe, fs, window):
    features_df = pd.DataFrame(columns=['meanEda',
                                        'stdEda',
                                        'kurtEda',
                                        'skewEda',
                                        'meanDerivative',
                                        'meanNegativeDerivative',
                                        'activity',
                                        'mobility',
                                        'complexity',
                                        'peaksCount',
                                        'meanPeakAmplitude',
                                        'meanRiseTime',
                                        'sumPeakAmplitude',
                                        'sumRiseTime',
                                        'sma',
                                        'energy',
                                        'varSpectralPower',
                                        'totalEnergyWavelet'
                                        ])  # add long dicts

    levels = 4
    n_mfcc = 20

    # create columns dynamically for the ones that returns more stuff
    # be careful if you modify this, the features MUST be in the right order

    # have to do like this because it saves the data sequentially, same for below
    for k in range(levels):
        features_df[f'energyWavelet_{k}'] = None
    for k in range(levels):
        features_df[f'distributionEnergy_{k}'] = None
    for k in range(levels):
        features_df[f'entropyWavelet_{k}'] = None

    for k in range(n_mfcc):
        features_df[f'meanMFCCS_{k}'] = None
    for k in range(n_mfcc):
        features_df[f'stdMFCCS_{k}'] = None
    for k in range(n_mfcc):
        features_df[f'medianMFCCS_{k}'] = None
    for k in range(n_mfcc):
        features_df[f'kurtMFCCS_{k}'] = None
    for k in range(n_mfcc):
        features_df[f'skewMFCCS_{k}'] = None

    features_df.drop(features_df.index, inplace=True)  # dropping the temporary None values I added to create
    # the columns

    for i in tqdm(range(0, dataframe.shape[0])):

        eda = dataframe[i, 1:]
        m, wd, eda_clean = process_statistical(eda, use_scipy=True, sample_rate=fs, new_sample_rate=fs,
                                               segment_width=window, segment_overlap=0)
        eda_clean = eda_clean[0]
        fh, xh = spectrum(eda, fs)
        derivative = np.gradient(eda_clean)
        secondDerivative = np.gradient(derivative)
        eda_phasic = wd["phasic_gsr"][0]

        peaks, amps, onsets = find_phasic_eda_peaks(eda_phasic)
        # EDA
        meanEda = np.mean(eda)  # one
        stdEda = np.std(eda)  # one
        kurtEda = scipy.stats.kurtosis(eda)  # one
        skewEda = scipy.stats.skew(eda)  # one

        meanDerivative = np.mean(derivative)  # one
        negativeDerivative = [i for i in derivative if i < 0]
        meanNegativeDerivative = np.mean(negativeDerivative)  # one

        activity = np.sum((eda - np.mean(eda)) ** 2)  # one
        mobility = np.sqrt(np.var(derivative) / np.var(eda))  # one
        complexity = np.sqrt(np.var(secondDerivative) / np.var(derivative)) / mobility  # one

        # Phasic
        riseTime = (peaks.squeeze() - onsets) / fs
        peaksCount = peaks.shape[0]  # one
        meanPeakAmplitude = np.mean(amps)  # one
        meanRiseTime = np.mean(riseTime)  # one
        sumPeakAmplitude = np.sum(amps)  # one
        sumRiseTime = np.sum(riseTime)  # one

        # Frequency
        sma = np.sum(np.abs(eda))  # one
        energy = np.sum(np.abs(eda) ** 2)  # one
        powerRange = fh[xh < 1]
        bandPowerIdx, _ = find_peaks(powerRange, height=0.01)
        matrix = np.array([powerRange[bandPowerIdx], bandPowerIdx])
        sortedMatrix = matrix[:, matrix[0, :].argsort()[::-1]]
        sortedFh = sortedMatrix[0, :]
        if len(sortedFh) < 5:
            m = 5 - len(sortedFh)
            padd = np.zeros((m))
            sortedFh = np.hstack((sortedFh, padd))
        bandPower = sortedFh[:5]
        # minSpectralPower = np.min(bandPower)
        # maxSpectralPower = np.max(bandPower)
        varSpectralPower = np.var(bandPower)  # one (np.float)

        # DWT Wavelets
        dwav = pywt.Wavelet('db3')
        dwtCoeffs = pywt.wavedec(eda_clean, wavelet=dwav, level=levels)
        detailedCoeff = dwtCoeffs[1:]
        energyWavelet = np.array([Energy(detailedCoeff, i) for i in range(levels)])  # array len(levels)
        totalEnergyWavelet = np.sum(energyWavelet)  # one
        distributionEnergy = np.array(
            [100 * energyWavelet[i] / totalEnergyWavelet for i in range(levels)])  # array len(levels)
        entropyWavelet = np.array([Entropy(energyWavelet[i]) for i in range(levels)])  # array len(levels)

        # MFCC
        # n_mfcc = 20 already set before loop
        mfccs = feature.mfcc(eda, sr=fs, n_mfcc=n_mfcc)
        meanMFCCS = np.mean(mfccs, axis=-1)  # n_mfcc
        stdMFCCS = np.std(mfccs, axis=-1)  # n_mfcc
        medianMFCCS = np.median(mfccs, axis=-1)  # n_mfcc
        kurtMFCCS = scipy.stats.kurtosis(mfccs, axis=-1)  # n_mfcc
        skewMFCCS = scipy.stats.skew(mfccs, axis=-1)  # n_mfcc

        what_to_stack = (meanEda, stdEda, kurtEda, skewEda, meanDerivative,
                         meanNegativeDerivative, activity, mobility, complexity,
                         peaksCount, meanPeakAmplitude, meanRiseTime,
                         sumPeakAmplitude, sumRiseTime, sma, energy,
                         varSpectralPower, totalEnergyWavelet, energyWavelet,
                         distributionEnergy, entropyWavelet, meanMFCCS, stdMFCCS,
                         medianMFCCS, kurtMFCCS, skewMFCCS)

        eda_features = np.hstack(what_to_stack)
        features_df.loc[i] = eda_features

    # now it returns a pd Dataframe
    return features_df


def extract_emg_featues(dataframe, fs):
    features_df = pd.DataFrame(columns=['rmse',
                                        'mav',
                                        'var',
                                        'energy',
                                        'mnf',
                                        'mdf',
                                        'zc',
                                        'fr',
                                        'mav_arr_0',
                                        'mav_arr_1',
                                        'mav_arr_2',
                                        'mav_arr_3',
                                        'std_0',
                                        'std_1',
                                        'std_2',
                                        'std_3',
                                        ])
    for i in tqdm(range(0, dataframe.shape[0])):
        emg = dataframe[i, 1:]
        emg = nk.emg_clean(emg, sampling_rate=fs)
        fh, xh = spectrum(emg, fs)
        N = len(emg)

        # Features
        rmse = np.sqrt(1 / N * np.sum(emg ** 2))
        mav = 1 / N * np.sum(np.abs(emg))
        var = 1 / (N - 1) * np.sum(emg ** 2)
        energy = np.sum(np.abs(emg) ** 2)
        mnf = MNF(fh, xh)
        mdf = MDF(fh, xh)
        zc = ZC(emg)
        fr = frequency_ratio(fh, xh, mnf)

        # DWT
        levels = 4
        dwav = pywt.Wavelet('db3')
        dwtCoeffs = pywt.wavedec(emg, wavelet=dwav, level=levels)
        detailedCoeff = dwtCoeffs[1:]
        mav_arr = np.array([1 / len(detailedCoeff[i]) * np.sum(np.abs(detailedCoeff[i])) for i in range(levels)])
        std = np.array([Std(detailedCoeff, mav_arr, i) for i in range(levels)])

        # added this to clean a bit
        what_to_stack = (rmse,
                         mav,
                         var,
                         energy,
                         mnf,
                         mdf,
                         zc,
                         fr,
                         mav_arr[0],
                         mav_arr[1],
                         mav_arr[2],
                         mav_arr[3],
                         std[0],
                         std[1],
                         std[2],
                         std[3])

        emg_features = np.hstack(what_to_stack)
        features_df.loc[i] = emg_features

    # now it returns a pd Dataframe
    return features_df


# Returns in a dictionary all the signal acquisitions in different DataFrames, indexed by file name
def extract_plux_data_basic(load_path):
    data_dict = {}
    if os.path.isfile(load_path):
        filenames = [load_path]
    else:
        filenames, _ = import_filenames(load_path)

    filenames = [name for name in filenames if '.txt' in name]
    for file in tqdm(filenames):
        signals_raw, fs = import_opensignals(file)
        std = np.std(signals_raw)
        mean = np.mean(signals_raw)
        mean[0] = 0
        std[0] = 1
        signals_norm = (signals_raw - mean) / std
        data_dict.update({file: signals_norm})

    return data_dict, fs


# Extracts data from Plux acquisition, and splits them in windows of window_size_sec seconds
def extract_plux_data_windowed(load_path, overlap_pct, window_size_sec, signal_name):
    filenames, dirs = import_filenames(load_path)
    filenames = [name for name in filenames if '.txt' in name]
    k = 0
    for file in tqdm(filenames):
        filepath = os.path.join(load_path, file)
        signals_raw, fs = import_opensignals(filepath)
        if k == 0:
            window_size = window_size_sec * fs
            final_set = np.zeros((1, window_size + 1))
        signal_raw = signals_raw[signal_name]
        std = np.std(signal_raw)
        mean = np.mean(signal_raw)
        signal_norm = (signal_raw - mean) / std
        windowed_signal = make_window(signal_norm.values, fs, overlap_pct, window_size_sec)
        key = np.repeat(np.array([k]), windowed_signal.shape[0], axis=0)
        key = np.expand_dims(key, axis=1)
        signal_set = np.hstack((key, windowed_signal))
        final_set = np.vstack((final_set, signal_set))
        k = k + 1

    return final_set[1:, :]


def find_phasic_eda_peaks(eda_phasic, percent=0.05):
    derivative = np.gradient(eda_phasic)
    df = signal_smooth(derivative, kernel="bartlett", size=20)
    # Zero crossings
    pos_crossings = signal_zerocrossings(df, direction="positive")
    neg_crossings = signal_zerocrossings(df, direction="negative")

    if pos_crossings[0] > neg_crossings[0]:
        neg_crossings = neg_crossings[1:]

    # Sanitize consecutive crossings
    if len(pos_crossings) > len(neg_crossings):
        pos_crossings = pos_crossings[0:len(neg_crossings)]
    elif len(pos_crossings) < len(neg_crossings):
        neg_crossings = neg_crossings[0:len(pos_crossings)]

    peaks_list = []
    onsets_list = []
    amps_list = []
    for i, j in zip(pos_crossings, neg_crossings):
        window = eda_phasic[i:j]
        amp = np.max(window)

        # Detected SCRs with amplitudes less than 10% of max SCR amplitude will be eliminated
        diff = amp - eda_phasic[i]
        if not diff < (percent * amp):
            peaks = np.where(eda_phasic == amp)[0]
            peaks_list.append(peaks)
            onsets_list.append(i)
            amps_list.append(amp)

    # Sanitize
    peaks = np.array(peaks_list)
    amps = np.array(amps_list)
    onsets = np.array(onsets_list)

    return peaks, amps, onsets


def spectrum(sample, fs):
    N = len(sample // 2)
    yf = fft(sample)
    xf = fftfreq(len(sample), 1 / fs)[:N // 2]
    fh = 2.0 / N * np.abs(yf[0:N // 2])
    return fh, xf


def MNF(fh, xh):
    powerSum = np.sum(fh)
    weightedPower = np.sum(xh * fh)
    mnf = weightedPower / powerSum
    return mnf


def MDF(fh, xh):
    matrix = np.array([fh, xh])
    sortedMatrix = matrix[:, matrix[0, :].argsort()]
    sortedFh = sortedMatrix[0, :]
    idx = equilibrium_index(sortedFh)
    mdf = xh[idx]
    return mdf


def ZC(sample):
    zc = np.sum(sample[:-1] * sample[1:] < 0)
    return zc


def frequency_ratio(fh, xh, mnf):
    idx = np.nonzero(xh == np.round(mnf))[0][0]
    lowFrequencies = np.sum(fh[:idx])
    highFrequencies = np.sum(fh[idx + 1:])
    fr = lowFrequencies / highFrequencies
    return fr


def equilibrium_index(sortedArray):
    diff = np.empty(len(sortedArray))
    for i in range(len(sortedArray)):
        diff[i] = np.abs((np.sum(sortedArray[0:i]) - np.sum(sortedArray[i + 1:])))
    equilibriumIdx = np.argmin(diff)
    return equilibriumIdx


def Energy(coeffs, k):
    return np.sum(np.array(coeffs[-k]) ** 2)


def Entropy(energy):
    return -np.sum(energy * np.log(energy))


def Std(coeff, mav, i):
    return np.sqrt(1 / (len(coeff[i] - 1)) * np.sum(np.abs(coeff[i] - mav[i]) ** 2))
