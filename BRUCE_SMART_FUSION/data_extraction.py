import pickle
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

def extract_data (path, srate, window_length):
    '''
    Inputs --  path - path of the data.
               srate - Sampling rate
               window_length - Length of the window, 32*srate in this case 
    Outputs -- Dictionary containing the infomation related to ECG, ACC, RESP signal.
    Description -- Function returns a dictionary which contains the ECG, ACC, RESP of every subject in
                PPG dalia dataset. Under dictionary with ECG as a key data contains ECG data in 32*srate 
               number of samples in one window it contain rpeaks and rpeak amplitude and data 
               according to different activities. Under ACC and RESP as key  ACC data in 32*srate
               segments and contains the data according to different activities.
    '''
    #new_changes
    subjects = [i for i in sorted(os.listdir(path)) if not(i.endswith('pdf'))] 
    seconds_per_window = window_length / srate      
    data = {}
    #index_1 = 0
    for sub_id in tqdm(subjects):
        #index_1+=1
        print('Subject Id is', sub_id) 
        windowed_ecg = []
        windowed_resp = []
        windowed_acc = []
        acc_x = []
        acc_y = []
        acc_z = []
        windowed_acc_x = []
        windowed_acc_y = []
        windowed_acc_z = []
        #import pdb;pdb.set_trace()
        subpath = os.path.join(path , sub_id , sub_id+ ".pkl")
        subpath_activity = pd.read_csv(os.path.join(path , sub_id , sub_id+ "_activity.csv")) 
        subpath_activity = subpath_activity.rename(columns = {'# SUBJECT_ID':'subject_id'}) 
        subpath_activity['subject_id'] = subpath_activity.iloc[:,0].astype('category') 
        subpath_activity['activity_id'] = subpath_activity.subject_id.cat.codes 
        start_time = subpath_activity.iloc[: , 1].values 
        ### Obtaining activity annotation as a list ### 
        for index in range(1,len(subpath_activity)):
            if index == 1:
                annotation_per_window = [subpath_activity.iloc[index-1,2] for i in range(int(round(subpath_activity.iloc[index,1] / seconds_per_window)))]
                prev = round(subpath_activity.iloc[index,1] / seconds_per_window) * seconds_per_window 
            else:
                annotation_per_window += [subpath_activity.iloc[index-1,2] for i in range(int(round((subpath_activity.iloc[index,1] - prev) / seconds_per_window)))]
                prev = round(subpath_activity.iloc[index,1] / seconds_per_window) * seconds_per_window 
        with open (subpath , 'rb') as f:
            data_dict = pickle.load(f , encoding='bytes')
        ECG = data_dict['ECG']
        RESP = data_dict['RESP']
        acc_data_x = data_dict['ACC']['ACC_X']
        acc_data_y = data_dict['ACC']['ACC_Y']
        acc_data_z = data_dict['ACC']['ACC_Z']
        rpeaks = data_dict['RPEAKS']
        #for item in acc_data:
        #    acc_x.append(item[0])
        #    acc_y.append(item[1])
        #    acc_z.append(item[2])
        acc_x_axis = np.array(acc_data_x)
        acc_y_axis = np.array(acc_data_y)
        acc_z_axis = np.array(acc_data_z)
        #import pdb;pdb.set_trace()
        #if model_type == 'raw_DL':
        #    ACC = np.concatenate((acc_y_axis.reshape(-1,1), acc_z_axis.reshape(-1,1)),axis= 0)
        #else:
        ACC = acc_data_y + acc_data_z
        ECG = ECG.flatten()
        RESP = RESP.flatten()
        len_parameter = int(np.round(len(ECG)/window_length))
        RPEAKS = [np.array([]) for i in range (len_parameter)]
        amplitudes = [np.array([]) for i in range (len_parameter)] 
        for i in range(len_parameter):
            windowed_ecg.append(ECG[i*window_length : (i+1)*window_length])
            windowed_resp.append(RESP[i*window_length : (i+1)*window_length])
            windowed_acc.append(ACC[i*window_length : (i+1)*window_length])
            windowed_acc_x.append(acc_x_axis[i * window_length : (i + 1) * window_length])
            windowed_acc_y.append(acc_y_axis[i * window_length : (i + 1) * window_length])
            windowed_acc_z.append(acc_z_axis[i * window_length : (i + 1) * window_length])
            for item in rpeaks:
                if item >= i*window_length and item < (i+1)*window_length:
                    sub_factor = i*window_length
                    item1 = ECG[item]
                    RPEAKS[i] = np.append(RPEAKS[i] , item - sub_factor)
                    amplitudes[i] = np.append(amplitudes[i] , item1)
        while len(annotation_per_window)!= len(windowed_ecg):
            annotation_per_window.append(1)
        data.update({sub_id : {'ECG' : {'ECG_DATA' : windowed_ecg , 'RPEAKS': RPEAKS , 'AMPLITUDES': amplitudes}
                         ,'ACC':{'ACC_DATA': windowed_acc, 'ACC_X':windowed_acc_x, 'ACC_Y':windowed_acc_y, 'ACC_Z': windowed_acc_z}
                          ,  'RESP': {'RESP_DATA': windowed_resp}
                          ,'ACTIVITY_ID': annotation_per_window}})
        #if index_1 == 1:
        #   break 
    return data
    
# path = 'C:/Users/ee19s/Desktop/BR_Uncertainty/FINAL_JOURNAL_DATA'
# srate = 256
# win_len = 32*srate
# key_id = 'S12'
# data = extract_data(path , srate , win_len)
# print(data[key_id].keys())
# print(len(data[key_id]['ECG']['ECG_DATA']))
# print(len(data[key_id]['ACC']['ACC_DATA']))
# print(len(data[key_id]['RESP']['RESP_DATA']))
# print(len(data[key_id]['ECG']['RPEAKS']))
# print(len(data[key_id]['ACTIVITY_ID']))
