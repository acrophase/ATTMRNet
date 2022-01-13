import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

subject = 'S17'
path = 'C:/Users/ee19s/Desktop/BR_Uncertainty/JOURNAL_DATA'
subject_path = os.path.join(path,subject)
data = pd.read_csv(os.path.join(subject_path+'/' + subject+ "_activity.csv"))
#data = pd.read_csv('C:/Users/ee19s/Desktop/BR_Uncertainty/JOURNAL_DATA/S1/S1_activity.csv')
print(data)


def sec_cov(time_info):
    h,m,s = time_info.split(':')
    sec_time = int(int(h)*3600 + int (m)*60 + float(s))
    return sec_time


time_stamps = data['Time_Stamps'].values
sec_array = np.array([])

for item in time_stamps:
    sec_array = np.append(sec_array , int (sec_cov(item)))
#print(sec_array)
sec_array = sec_array - sec_array[0]
data['second_stamp'] = sec_array.reshape(-1,1)
convert_dict = {'second_stamp': int} 
data = data.astype(convert_dict)
print(data)
data.to_csv(os.path.join(subject_path+'/' + subject+ "_activity.csv"))

