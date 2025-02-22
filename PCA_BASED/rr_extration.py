import numpy as np
import scipy
from data_extraction import extract_data 
from resp_signal_extraction import *

def extremas_extraction(signal):
    '''
    Input --  Respiratory signal
    Output -- Average breathing duration and relevent extremas.

    Description -- This function takes the respiratory signal as an argument
                  and then by using count advance algorithm to detect the
                  breathing cycle based on maximas and minimas.
                  For more details refer--Schäfer, A., & Kratky, K. W. (2008). 
                  Estimation of breathing rate from respiratory sinus arrhythmia: 
                  Comparison of various methods.Annals of Biomedical Engineering,
                  36(3), 476–485. https://doi.org/10.1007/s10439-007-9428-1.

                  Based on this algorithm this function return the average breathing duration
                  and relevent extremas
    '''
    avg_breath_duration = np.array([])
    extrema_relevent = []
    for item in signal:
        print("I am running")
        amplitude = np.array([])
        pos_peaks , _ = scipy.signal.find_peaks(item , height = [-3000,3000])
        neg_peaks , _ = scipy.signal.find_peaks(-1*item , height = [-3000 , 3000])
        extremas = np.concatenate((pos_peaks , neg_peaks))
        extremas = np.sort(extremas)
        for i in range(len(extremas)):
            amplitude = np.append(amplitude , item[int(extremas[i])])
        amplitude_diff = np.abs(np.diff(amplitude))
        q3 = np.percentile(amplitude_diff , 75)
        threshold = 0.3*q3
        eliminate_pairs_of_extrema = 1
        while(eliminate_pairs_of_extrema):
            amps = np.array([])
            if len(extremas)<3:
                eliminate_pairs_of_extrema = 0
                continue
            for i in range(len(extremas)):
                amps = np.append(amps , item[int(extremas[i])])
            amp_diff = np.abs(np.diff(amps)) 
            min_amp_diff , index = min(amp_diff) , np.argmin(amp_diff)
            #print(min_amp_diff)
            if min_amp_diff > threshold:
                eliminate_pairs_of_extrema = 0
                #extrema_relevent = extremas
            else:
                extremas = np.concatenate((extremas[0:index] , extremas[index+2 :]))
                #amplitude_diff = np.delete(amplitude_diff , index)
        if(len(extremas) > 2):
            if item[int(extremas[0])] < item[int(extremas[1])]:
                extremas = extremas[1:]
            if item[int(extremas[-1])] < item[int(extremas[-2])]:
                extremas = extremas[:-1]
        no_of_breaths = (len(extremas)-1)/2
        breath_duration = extremas[-1] - extremas[0]
        if(no_of_breaths != 0):
            avg_breath_duration = np.append(avg_breath_duration , breath_duration/no_of_breaths)
        extrema_relevent.append(extremas)
    return avg_breath_duration , extrema_relevent     

# srate = 700
# path = '/media/acrophase/pose1/charan/BR_Uncertainty/ppg_dalia_data'
# win_len = 32*srate
# data = extract_data(path,srate,win_len)

# acc_x = data['S14']['ACC']['ACC_X']
# acc_y = data['S14']['ACC']['ACC_Y']
# acc_z = data['S14']['ACC']['ACC_Z']

# resp_sig = resp_signal_extract(acc_x,acc_y,acc_z)

# avg_br , _ = extremas_extraction(resp_sig)
# rr = (60*srate)/avg_br

# print(rr)
# print(len(rr))
# print(len(acc_x))