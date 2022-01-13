import pandas as pd
import matplotlib.pyplot as plt
from filters import *
import scipy
from scipy.signal import find_peaks
import numpy as np
import neurokit2 as nk
from hrv_analysis.extract_features import _create_interpolation_time, _create_time_info
import pickle as pkl
import os 
import sys

# srate = 256
# #nyquist = srate/2
# #frange  = [5,15]
# #fkernB,fkernA = scipy.signal.cheby2(6,30,np.array(frange)/nyquist,btype='bandpass')

# subject = 'S17'
# path = 'C:/Users/ee19s/Desktop/BR_Uncertainty/JOURNAL_DATA'
# subject_path = os.path.join(path,subject)
# # ecg = pd.read_csv(os.path.join(subject_path+'/' + subject_small+ "_ECGmV.csv"))
# # resp = pd.read_csv(os.path.join(subject_path+'/' + subject_small + "_Belt.csv"))
# # acc_large = pd.read_csv(os.path.join(subject_path+'/' + subject_small + "_AccmG_HR.csv"))

# with open(os.path.join(subject_path+ '/'+subject+'.pkl') , 'rb') as f:
#     data = pkl.load(f)

# ecg = data['ECG']
# resp = data['RESP']
# rpeaks = data['RPEAKS']
# acc_x = data['ACC']['ACC_X']
# acc_y = data['ACC']['ACC_Y']
# acc_z = data['ACC']['ACC_Z']

# # _, rpeaks_ecg1 = nk.ecg_peaks(ecg, sampling_rate = srate/1.5)
# # rpeaks = rpeaks_ecg1['ECG_R_Peaks']
# # rpeaks1_amp = ecg[rpeaks]


# # resp = resp[3500:len(resp)-2500]
# # resp = scipy.signal.resample(resp , (len(ecg)))
# # rpeaks = np.append(rpeaks , np.array([1439]))
# # np.sort(rpeaks)
# # false_peak = np.asarray([1439,139020,154754,426236,472673])
# # false_pk_loc = []
# # for i in false_peak:
# #     false_pk_loc.append(np.where(rpeaks == i)[0])
# # rpeaks = np.delete(rpeaks , false_pk_loc)
# # rpeaks1_amp = ecg[rpeaks]

# # index_ecg1 = np.where(np.logical_or(rpeaks1_amp < 0.25, rpeaks1_amp>7))[0]
# # new_peak_loc1 = np.delete(rpeaks,index_ecg1)

# print(len(ecg))
# print(len(resp))
# print(len(acc_x))
# print(len(acc_y))
# print(len(acc_z))

# plt.plot(ecg)
# plt.plot(rpeaks , ecg[rpeaks] , 'r*')
# plt.grid(True)
# plt.show()

# # plt.plot(resp)
# # plt.grid(True)
# # plt.show()

# # plt.plot(acc_x)
# # plt.grid(True)
# # plt.show()

# # plt.plot(acc_y)
# # plt.grid(True)
# # plt.show()

# # plt.plot(acc_z)
# # plt.grid(True)
# # plt.show()

# # data = {'ECG':ecg, 'RPEAKS':rpeaks,'RESP':resp
# #                 ,'ACC':{'ACC_X':acc_x,'ACC_Y':acc_y,'ACC_Z':acc_z} }

# # with open(os.path.join(subject_path+'/'+subject+'.pkl'), 'wb') as handle:
# #     pkl.dump(data, handle)

annotation = pd.read_pickle('C:/Users/ee19s/Desktop/BR_Uncertainty/BRUCE_DATASET_CODE/annotation.pkl')

with open('output','rb') as f:
    output_data = pkl.load(f)

with open('input','rb') as f:
    input_data = pkl.load(f)

with open('raw_signal.pkl','rb') as f:
    raw_data = pkl.load(f)

print(input_data.shape)
print(output_data.shape)
print(raw_data.shape)
print(annotation.shape)

plt.plot(raw_data[1028][2])
plt.grid(True)
plt.show()

