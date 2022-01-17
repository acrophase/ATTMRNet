import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal
import pywt
from scipy import interpolate
from scipy import fft
import pickle
import pandas as pd
from data_extraction import extract_data 
from filters import *
from hrv_analysis.extract_features import _create_interpolation_time, _create_time_info

srate = 256
fbpB , fbpA = band_pass(0.1,0.8,8)
fkern_lp_B ,fkern_lp_A = cheby_lp(6,25,0.8)              
fkern_hp_B , fkern_hp_A = cheby_hp(4,20,0.1)   

flpB_ref,flpA_ref = scipy.signal.cheby2(6,20 , 0.6/(srate/2),btype='lowpass')
fhpB_ref,fhpA_ref = scipy.signal.cheby2(4,20 ,0.1/(srate/2),btype='highpass')

def edr_adr_extraction(acc , rpeaks , rpeak_amplitudes,reference_resp, interp_rate = 4):
    '''
    inputs -- acc - Accelerometer signal extracted from dictionary returned by PPG_Dalia_data_extraction()
              rpeaks - R peak indices obtained from dictionary returned by PPG_Dalia_data_extraction()
              rpeak_amplitudes - R peak amplitudes obtained from dictionary returned by PPG_Dalia_data_extraction()
              reference_resp --  reference respiratory signal.
    
    outputs -- Function returns edr signals by HRV and RPEAK amplitude variations and ADR signal from accelerometer
               Final reference respiratory signal.
    Description -- Function takes the ACC, RPEAKS, Rpeak amplitudes for a particular subject and then calculate
                  the respiratory signal based of HRV, rpeak amplitude variations and using adaptive filtering
                  for accelerometer data. 
    '''
    final_edr_hrv = []
    final_edr_rpeak = []
    final_adr = []
    final_ref_resp = []
    eps = 0.0001
    #-----------------------RESPIRATORY SIGNAL BY HRV------------------------------------
    # interpolate the rr interval using cubic spline interpolation and filter between 
    # 0.1Hz - 1Hz to obtain final edr
    #cnt_1 = 0
    for item in rpeaks:
        #import pdb;pdb.set_trace()
        rr_interval = (np.diff(item)/srate)*1000
        index = np.where(rr_interval <= 0)
        rr_interval = np.delete(rr_interval , index)
        #_,unique_index = np.unique(rr_interval , return_index = True)
        #rr_interval = rr_interval[np.sort(unique_index)]
        index_sets = [np.argwhere(i==rr_interval) for i in np.unique(rr_interval)]
        dup_ind_set = [i for i in index_sets if (len(i)>1)]
        for sub_arr in dup_ind_set:
            cnt = 0
            for i in sub_arr:
                rr_interval[i[0]] = rr_interval[i[0]]+cnt*eps
                cnt+=1
        #cnt_1+=1
        #if(cnt_1 == 12):
        #    import pdb;pdb.set_trace()
        
        rr_times = _create_time_info(list(rr_interval))
        funct = interpolate.interp1d(x=rr_times, y=list(rr_interval), kind='cubic')
        timestamps_interpolation = _create_interpolation_time(rr_times, 4)
        interpolated_signal = funct(timestamps_interpolation)
        #time_stamp_hrv = np.arange(0,len(rr_interval))
        #time_interp_hrv = np.arange(time_stamp_hrv[0] , time_stamp_hrv[-1] , 1/interp_rate)
        #interpolated_signal = scipy.interpolate.griddata(time_stamp_hrv , rr_interval , time_interp_hrv , method='cubic')
        interpolated_signal = (interpolated_signal - np.mean(interpolated_signal))/np.std(interpolated_signal)
        filt_sig = scipy.signal.filtfilt(fbpB , fbpA , interpolated_signal,method = 'gust')
        #filt_sig = np.append(filt_sig , np.zeros(128 - len(filt_sig)))
        final_edr_hrv.append(filt_sig)
    #---------------------RESPIRATORY SIGNAL BY RPEAKS-----------------------------------
    # interpolate the rpeak amplitudes using cubic spline interpolation and filter between 
    # 0.1Hz - 1Hz to obtain final edr
    i = 0
    cnt_1 = 0
    for item in rpeak_amplitudes:
        #final_dup_index = np.array([])
        rr_interval = (np.diff(rpeaks[int(i)])/srate)*1000
        rr_interval = rr_interval.astype(float)
        index = np.where(rr_interval <= 0)
        rr_interval = np.delete(rr_interval , index)
        item = np.delete(item,index)
        #_,unique_index= np.unique(rr_interval , return_index = True)
        #import pdb;pdb.set_trace()
        index_sets = [np.argwhere(i==rr_interval) for i in np.unique(rr_interval)]
        dup_ind_set = [i for i in index_sets if (len(i)>1)]
        for arr in dup_ind_set:
            cnt = 0
            for ele in arr:
                rr_interval[int(ele[0])] = rr_interval[int(ele[0])] + (cnt*eps)
                cnt+=1
        #new_arr = np.delete(item , final_dup_index)
        #import pdb;pdb.set_trace()
        #cnt_1+=1
        #if(cnt_1 == 12):
        #    import pdb;pdb.set_trace()
        rr_times = _create_time_info(list(rr_interval))
        funct = interpolate.interp1d(x=rr_times, y=list(item[1:]), kind='cubic')
        timestamps_interpolation = _create_interpolation_time(rr_times, 4)
        interpolated_signal_rp = funct(timestamps_interpolation)
        #time_stamp_rpeak = np.arange(0 , len(item))
        #time_interp_rpeak = np.arange(time_stamp_rpeak[0] , time_stamp_rpeak[-1] , 1/interp_rate)
        #interpolated_signal_rp = scipy.interpolate.griddata(time_stamp_rpeak , item , time_interp_rpeak ,method='cubic' )
        interpolated_signal_rp = (interpolated_signal_rp - np.mean(interpolated_signal_rp))/np.std(interpolated_signal_rp)
        filt_sig = scipy.signal.filtfilt(fbpB,fbpA , interpolated_signal_rp, method = 'gust')
        #filt_sig = np.append(filt_sig , np.zeros(128 - len(filt_sig)))
        final_edr_rpeak.append(filt_sig)
        i+=1

    #-------------------------RESPIRATORY SIGNAL BY ACCELEROMETER-------------------------
    # calculate the fft of accelerometer data and then select the spectrum between
    # the frequency range of 0.1Hz - 1Hz the frequency correspond to the maximum
    # power will be taken as central frequency and then that will decide the 
    # lower cut off frequency or upper cuttoff frequency of the filter to obtain
    # the respiratory signal.
    j=0
    for item in acc:
        lp_filt_sig = scipy.signal.filtfilt(fkern_lp_B , fkern_lp_A , item)
        hp_filt_sig = scipy.signal.filtfilt(fkern_hp_B , fkern_hp_A , lp_filt_sig)
        spectrum = np.absolute(scipy.fft.fft(hp_filt_sig)**2)
        freq = scipy.fft.fftfreq(len(spectrum) , d= 1/srate)
        upper_index = int(len(item)/srate + 1)
        lower_index = int((0.1*len(item))/srate)
        rel_freq = freq[lower_index:upper_index]
        rel_spectrum = spectrum[lower_index:upper_index]
        max_freq = rel_freq[np.argmax(rel_spectrum)]
        lower_cut_freq = max(0.1 , max_freq-0.4)
        upper_cut_freq = max_freq + 0.4
        flpB ,flpA = scipy.signal.cheby2(5,30,upper_cut_freq/(srate/2) , btype='lowpass')
        fhpB , fhpA = scipy.signal.cheby2(4,30, lower_cut_freq/(srate/2) , btype='highpass')
        lp_filt_acc = scipy.signal.filtfilt(flpB, flpA , hp_filt_sig)
        final_signal = scipy.signal.filtfilt(fhpB , fhpA, lp_filt_acc)
        resample_sig = scipy.signal.resample(final_signal , len(final_edr_rpeak[j]))
        #resample_sig = np.append(resample_sig , np.zeros(128 - len(resample_sig)))
        final_adr.append(resample_sig)
        j+=1
    
    k = 0
    for item in reference_resp:
        lp_filt = scipy.signal.filtfilt(flpB_ref,flpA_ref , item)
        hp_filt = scipy.signal.filtfilt(fhpB_ref,fhpA_ref , lp_filt)
        resmp_signal = scipy.signal.resample(hp_filt , len(final_edr_rpeak[k]))
        #resmp_signal = np.append(resmp_signal , np.zeros(128 - len(resmp_signal)))
        final_ref_resp.append(resmp_signal)
        k+=1
    return final_edr_hrv ,final_edr_rpeak ,final_adr,final_ref_resp

# path = 'C:/Users/ee19s/Desktop/BR_Uncertainty/FINAL_JOURNAL_DATA'
# srate = 256
# win_len = 32*srate
# data = extract_data(path , srate , win_len)
# key_id = 'S21'

# rpeaks = data[key_id]['ECG']['RPEAKS']
# amps = data[key_id]['ECG']['AMPLITUDES']
# acc = data[key_id]['ACC']['ACC_DATA']
# resp = data[key_id]['RESP']['RESP_DATA']

# edr_hrv,edrpeak,adr,ref_resp = edr_adr_extraction(acc,rpeaks,amps,resp)

# plt.plot(edr_hrv[16])
# plt.grid(True)
# plt.title("EDR_HRV")
# plt.show()

# plt.plot(edrpeak[15])
# plt.grid(True)
# plt.title("EDR_RPEAK")
# plt.show()

# plt.plot(adr[15])
# plt.grid(True)
# plt.title("ADR")
# plt.show()

# plt.plot(ref_resp[15])
# plt.grid(True)
# plt.title("REF RESP")
# plt.show()