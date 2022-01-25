import numpy as np
import scipy
from scipy import interpolate
from scipy import fft
from data_extraction import extract_data 
from resp_signal_extraction import edr_adr_extraction

def rr_extraction(resp_signal, srate):
    """
    input -- raw respiratory signals in the window of 32 secs.
             srate -- sampling rate.
    output -- returns the average breath duration for one segement
              and other morphology based parameters.
    Description -- This function takes the respiratory signal as an argument
                  and then by using count advance algorithm to detect the
                  breathing cycle based on maximas and minimas.
                  For more details refer--Schäfer, A., & Kratky, K. W. (2008).
                  Estimation of breathing rate from respiratory sinus arrhythmia:
                  Comparison of various methods.Annals of Biomedical Engineering,
                  36(3), 476–485. https://doi.org/10.1007/s10439-007-9428-1.

                  It also return the time domain features related to peaks.
                  like coefficient_of_var, mean_peak_to_peak amplitude ,
                  true_max_true_min , coefficient of variation of minima.
    """
    avg_breath_duration = np.array([])
    coefficient_of_var = []
    mean_p_to_p = []
    true_max_true_min = []
    coefficient_of_var_minima = []
    for item in resp_signal:
        amplitude = np.array([])
        rel_minima = np.array([])
        pos_peaks, _ = scipy.signal.find_peaks(item, height=[-3000, 3000])
        neg_peaks, _ = scipy.signal.find_peaks(-1 * item, height=[-3000, 3000])
        extremas = np.concatenate((pos_peaks, neg_peaks))
        extremas = np.sort(extremas)
        amplitude = item[extremas]
        # for i in range(len(extremas)):
        #    amplitude = np.append(amplitude , item[int(extremas[i])])
        amplitude_diff = np.abs(np.diff(amplitude))
        q3 = np.percentile(amplitude_diff, 75)
        threshold = 0.3 * q3
        eliminate_pairs_of_extrema = 1
        while eliminate_pairs_of_extrema:
            # amps = np.array([])
            if len(extremas) < 3:
                eliminate_pairs_of_extrema = 0
                continue
            # for i in range(len(extremas)):
            #    amps = np.append(amps , item[int(extremas[i])])
            amps = item[extremas]
            amp_diff = np.abs(np.diff(amps))
            min_amp_diff, index = min(amp_diff), np.argmin(amp_diff)
            if min_amp_diff > threshold:
                eliminate_pairs_of_extrema = 0
            else:
                extremas = np.concatenate((extremas[0:index], extremas[index + 2 :]))
        if item[int(extremas[0])] < item[int(extremas[1])]:
            extremas = extremas[1:]
        if item[int(extremas[-1])] < item[int(extremas[-2])]:
            extremas = extremas[:-1]
        no_of_breaths = (len(extremas) - 1) / 2
        breath_duration = extremas[-1] - extremas[0]
        avg_breath_duration = np.append(
            avg_breath_duration, breath_duration / no_of_breaths
        )
        amps_relevent = item[extremas]
        peak_to_peak_amp = np.abs(np.diff(amps_relevent))
        coefficient_of_var.append(np.std(peak_to_peak_amp) / np.mean(peak_to_peak_amp))
        mean_p_to_p.append(np.mean(peak_to_peak_amp))
        true_max_true_min.append((len(extremas)) / (len(pos_peaks) + len(neg_peaks)))
        for i in extremas:
            if i in neg_peaks:
                rel_minima = np.append(rel_minima, i)
        time_diff = np.diff((rel_minima) / (srate))
        coefficient_of_var_minima.append(np.std(time_diff) / np.mean(time_diff))
    return (
        avg_breath_duration,
        coefficient_of_var,
        mean_p_to_p,
        true_max_true_min,
        coefficient_of_var_minima,
    )

# path = '/media/acrophase/pose1/charan/BR_Uncertainty/BRUCE_DATA_SET/FINAL_JOURNAL_DATA'
# srate = 256
# win_len = 32*srate
# data = extract_data(path , srate , win_len)
# print(data.keys())
# key_id = 'S21'


# rpeaks = data[key_id]['ECG']['RPEAKS']
# amps = data[key_id]['ECG']['AMPLITUDES']
# acc = data[key_id]['ACC']['ACC_DATA']
# resp = data[key_id]['RESP']['RESP_DATA']

# edr_hrv,edr_rpeak,adr = edr_adr_extraction(acc,rpeaks,amps,resp)
# bd_hrv ,cv_hrv,mpp_hrv,tm_hrv, cvm_hrv = rr_extraction(edr_hrv,srate)
# bd_rpeak ,cv_rpeak,mpp_rpeak,tm_rpeak, cvm_rpeak = rr_extraction(edr_rpeak,srate)
# bd_adr,cv_adr,mpp_adr,tm_adr, cvm_adr = rr_extraction(adr,srate)
# #bd_ref_resp , cv_ref,mpp_ref,tm_ref, cvm_ref= rr_extraction(ref_resp,srate)

# rr_hrv = (60*4)/bd_hrv
# rr_rpeak = (60*4)/bd_rpeak
# rr_adr = 60/(bd_adr/(srate/10))
# # rr_ref = (60*4)/bd_ref_resp

# print(len(rr_hrv))
# print(len(rr_rpeak))
# print(len(rr_adr))
# #print(len(rr_ref))
# print("==========================================================================================")
# print(len(cv_hrv))
# print(len(cv_rpeak))
# print(len(cv_adr))
# #print(len(cv_ref))
# print("============================================================================================")
# print(len(mpp_hrv))
# print(len(mpp_rpeak))
# print(len(mpp_adr))
# #print(len(mpp_ref))
# print("=============================================================================================")
# print(len(tm_hrv))
# print(len(tm_rpeak))
# print(len(tm_adr))
# #print(len(tm_ref))
# print("==============================================================================================")
# print(len(cvm_hrv))
# print(len(cvm_rpeak))
# print(len(cvm_adr))
# #print(rr_adr)
#print(len(cvm_ref))