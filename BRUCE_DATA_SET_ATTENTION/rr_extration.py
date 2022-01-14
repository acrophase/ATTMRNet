import numpy as np
import scipy
from data_extraction import extract_data 
from resp_signal_extraction import edr_adr_extraction

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
        amplitude = np.array([])
        pos_peaks , _ = scipy.signal.find_peaks(item , height = [-300,300])
        neg_peaks , _ = scipy.signal.find_peaks(-1*item , height = [-300 , 300])
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

# path = 'C:/Users/ee19s/Desktop/BR_Uncertainty/FINAL_JOURNAL_DATA'
# srate = 256
# win_len = 32*srate
# data = extract_data(path , srate , win_len)
# key_id = 'S21'


# rpeaks = data[key_id]['ECG']['RPEAKS']
# amps = data[key_id]['ECG']['AMPLITUDES']
# acc = data[key_id]['ACC']['ACC_DATA']
# resp = data[key_id]['RESP']['RESP_DATA']

# edr_hrv,edr_peak,adr,ref_resp = edr_adr_extraction(acc,rpeaks,amps,resp)
# #import pdb;pdb.set_trace()
# extrema_hrv , _ = extremas_extraction(edr_hrv)
# extrema_rpeak , _ = extremas_extraction(edr_peak)
# extrema_adr,_ = extremas_extraction(adr)
# extrema_ref_resp , _ = extremas_extraction(ref_resp)

# rr_hrv = (60*4)/extrema_hrv
# rr_rpeak = (60*4)/extrema_rpeak
# rr_adr = (60*4)/extrema_adr
# rr_ref_resp = (60*4)/extrema_ref_resp

# print("============================================HRV RR==================================================")
# while(len(rr_hrv) != len(rr_ref_resp)):
#     if(len(rr_hrv) > len(rr_ref_resp)):
#         index  = np.argmax(rr_hrv)
#         rr_hrv = np.delete(rr_hrv , index)
#     else:
#         index = np.argmax(rr_ref_resp)
#         rr_ref_resp = np.delete(rr_ref_resp , index)

# error_hrv = np.absolute(rr_hrv - rr_ref_resp)
# mae_hrv = np.mean(error_hrv)
# print(mae_hrv)
# print("============================================RPEAK RR==================================================")

# while(len(rr_rpeak) != len(rr_ref_resp)):
#     if(len(rr_rpeak) > len(rr_ref_resp)):
#         index  = np.argmax(rr_rpeak)
#         rr_rpeak = np.delete(rr_rpeak , index)
#     else:
#         index = np.argmax(rr_ref_resp)
#         rr_ref_resp = np.delete(rr_ref_resp , index)


# error_rpeak = np.absolute(rr_rpeak - rr_ref_resp)
# mae_rpeak = np.mean(error_rpeak)
# print(mae_rpeak)
# print("============================================ADR RR==================================================")

# while(len(rr_adr) != len(rr_ref_resp)):
#     if(len(rr_adr) > len(rr_ref_resp)):
#         index  = np.argmax(rr_adr)
#         rr_adr = np.delete(rr_adr , index)
#     else:
#         index = np.argmax(rr_ref_resp)
#         rr_ref_resp = np.delete(rr_ref_resp , index)


# error_adr = np.absolute(rr_adr - rr_ref_resp)
# mae_adr = np.mean(error_adr)
# print(mae_adr)


# print(len(rr_hrv))
# print(len(rr_rpeak))
# print(len(rr_adr))
# print(len(rr_ref_resp))

# print(len(data[key_id]['ACC']['ACC_X']))