import pandas as pd
import matplotlib.pyplot as plt
from filters import *
import scipy
import numpy as np
import neurokit2 as nk
from hrv_analysis.extract_features import _create_interpolation_time, _create_time_info
import pickle as pkl
import os 

srate = 256
#nyquist = srate/2
#frange  = [5,15]
#fkernB,fkernA = scipy.signal.cheby2(6,30,np.array(frange)/nyquist,btype='bandpass')

subject = 'SUDHANSHU'
subject_small = 'sudhanshu'
path = 'C:/Users/ee19s/Desktop/BR_Uncertainty/JOURNAL_DATA'
subject_path = os.path.join(path,subject)
ecg = pd.read_csv(os.path.join(subject_path+'/' + subject_small+ "_ECGmV.csv"))
resp = pd.read_csv(os.path.join(subject_path+'/' + subject_small + "_Belt.csv"))
acc_large = pd.read_csv(os.path.join(subject_path+'/' + subject_small + "_AccmG_HR.csv"))

#acc_small = pd.read_csv('C:/Users/ee19s/Desktop/BR_Uncertainty/JOURNAL_DATA/subject/amit_anand_AccMG.csv')

ecg1 = ecg['ECG Lead 1'].values
ecg2 = ecg['ECG Lead 2'].values
resp_data = resp['Breathing Wave'].values
acc_x= acc_large['Lateral Acc'].values
acc_z = acc_large['Longitudinal Acc'].values
acc_y = acc_large['Vertical Acc'].values
# ecg1 = ecg1[1400:1738778]
# acc_x = acc_x[1400:1738778]
# acc_y = acc_y[1400:1738778]
# acc_z = acc_z[1400:1738778]
# resp_data = resp_data[300:173560]
plt.plot(resp_data)
plt.grid(True)
plt.show()
#resamp_ecg1 = baseline_removal(ecg1)
#resamp_ecg2 = baseline_removal(ecg2)
#resamp_ecg1 = scipy.signal.filtfilt(fkernB,fkernA,resamp_ecg1)
#resamp_ecg2 = scipy.signal.filtfilt(fkernB,fkernA,resamp_ecg2)

_, rpeaks_ecg1 = nk.ecg_peaks(ecg1, sampling_rate = srate)
_, rpeaks_ecg2 = nk.ecg_peaks(ecg2, sampling_rate = srate)


rpeaks1_loc = rpeaks_ecg1['ECG_R_Peaks']
rpeaks1_amp = ecg1[rpeaks1_loc]

# rpeaks1_loc = np.append(rpeaks1_loc , np.array([11469,22950,26804,30645]))
# np.sort(rpeaks1_loc)
# false_peak = np.asarray([7557,15237,19081,22916,26758,30597,307077,560517])
# false_pk_loc = []
# for i in false_peak:
#     false_pk_loc.append(np.where(rpeaks1_loc == i)[0])
# rpeaks1_loc = np.delete(rpeaks1_loc , false_pk_loc)
# rpeaks1_amp = ecg1[rpeaks1_loc]

index_ecg1 = np.where(np.logical_or(rpeaks1_amp < 0.35, rpeaks1_amp>1.5))[0]
# #index_ecg2 = np.where(rpeaks1_amp>2)[0]
new_peak_loc1 = np.delete(rpeaks1_loc,index_ecg1)
new_peak_amp1 = np.delete(rpeaks1_amp,index_ecg1)
# new_peak_loc1 = rpeaks1_loc
# new_peak_amp1 = rpeaks1_amp

plt.plot(ecg1)
plt.plot(new_peak_loc1 , new_peak_amp1 , 'r*')
plt.grid(True)
plt.show()

resp_data = scipy.signal.resample(resp_data , (len(ecg1)))
acc_x = scipy.signal.resample(acc_x, len(ecg1))
acc_y = scipy.signal.resample(acc_y, len(ecg1))
acc_z = scipy.signal.resample(acc_z, len(ecg1))

print(len(ecg1))
print(len(acc_x))
print(len(resp_data))

# data = {'ECG':ecg1, 'RPEAKS':new_peak_loc1,'RESP':resp_data
#                 ,'ACC':{'ACC_X':acc_x,'ACC_Y':acc_y,'ACC_Z':acc_z} }

# with open(os.path.join(subject_path+'/'+'S21.pkl'), 'wb') as handle:
#     pkl.dump(data, handle)

# print(len(data['ECG']))
# print(len(data['RPEAKS']))
# print(len(data['RESP']))
# print(len(data['ACC']['ACC_X']))

# plt.plot(data['ECG'])
# #plt.plot(new_peak_loc1 , new_peak_amp1 , 'r*')
# plt.grid(True)
# plt.show()

# plt.plot(data['RESP'])
# plt.grid(True)
# plt.show()


