import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


ecg = pd.read_csv('C:/Users/ee19s/Desktop/BR_Uncertainty/JOURNAL_DATA/PRAJNA/prajna_ECGmV.csv')
resp = pd.read_csv('C:/Users/ee19s/Desktop/BR_Uncertainty/JOURNAL_DATA/PRAJNA/prajna_Belt.csv')
acc_large = pd.read_csv('C:/Users/ee19s/Desktop/BR_Uncertainty/JOURNAL_DATA/PRAJNA/prajna_AccmG_HR.csv')

acc_small = pd.read_csv('C:/Users/ee19s/Desktop/BR_Uncertainty/JOURNAL_DATA/PRAJNA/prajna_AccMG.csv')

ecg_data = ecg['ECG Lead 1'].values
resp_data = resp['Breathing Wave'].values
acc_lateral_large = acc_large['Lateral Acc'].values
acc_longitudinal_large = acc_large['Longitudinal Acc'].values
acc_vertical_large = acc_large['Vertical Acc'].values

acc_lateral_small = acc_small['Lateral Acc'].values
acc_longitudinal_small = acc_small['Longitudinal Acc'].values
acc_vertical_small = acc_small['Vertical Acc'].values
#resp_data = resp_data[0:152000]
#acc_lateral_small = acc_lateral_small[0:152000]
print(len(ecg_data))
print(len(resp_data))
print(len(acc_lateral_large))
print(len(acc_lateral_small))

plt.figure(1)
plt.title("ECG")
plt.plot(ecg_data)
plt.grid(True)
plt.show()

plt.figure(2)
plt.title("RESP")
plt.plot(resp_data)
plt.grid(True)
plt.show()

plt.figure(3)
plt.title("acc_lateral_large")
plt.plot(acc_lateral_large)
plt.grid(True)
plt.show()

plt.figure(4)
plt.title("acc_longitudinal_large")
plt.plot(acc_longitudinal_large)
plt.grid(True)
plt.show()

plt.figure(5)
plt.title("acc_vertical_large")
plt.plot(acc_vertical_large)
plt.grid(True)
plt.show()

plt.figure(6)
plt.title("acc_lateral_small")
plt.plot(acc_lateral_small)
plt.grid(True)
plt.show()

plt.figure(7)
plt.title("acc_longitudinal_small")
plt.plot(acc_longitudinal_small)
plt.grid(True)
plt.show()

plt.figure(8)
plt.title("acc_vertical_small")
plt.plot(acc_vertical_small)
plt.grid(True)
plt.show()