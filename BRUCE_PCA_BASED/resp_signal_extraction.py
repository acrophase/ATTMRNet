from data_extraction import extract_data 
import numpy as np
from filters import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle as pkl

fhp_eeB , fhp_eeA = high_pass(5,20,0.2)   
fhpB_high_ee , fhpA_high_ee = high_pass(5,20,0.3)
flpB_low_ee , flpA_low_ee = low_pass(5,25,0.4)
flpB_med_ee , flpA_med_ee = low_pass(6,25,0.7)
flpB_high_ee , flpA_high_ee = low_pass(6,25,0.8)

fkern_lpB , fkern_lpA = low_pass(6,35,0.7)
fkern_hpB , fkern_hpA = high_pass(4,20,0.1)

def resp_signal_extract(acc_x,acc_y,acc_z):
    final_resp_sig = [np.array([]) for i in range(len(acc_x))]
    cnt = 0
    for i in range(len(acc_x)):
        temp_x = acc_x[i]
        temp_y = acc_y[i]
        temp_z = acc_z[i]
        temp_diff_x = np.diff(acc_x[i])
        temp_diff_y = np.diff(acc_y[i])
        temp_diff_z = np.diff(acc_z[i])
        
        net_signal = np.sqrt((temp_diff_x**2) + (temp_diff_y**2) + (temp_diff_z**2))
        ee = np.sum(net_signal)
        
        if(ee<100):
            lp_filt_x = scipy.signal.filtfilt(flpB_low_ee , flpA_low_ee , temp_x)
            lp_filt_y = scipy.signal.filtfilt(flpB_low_ee , flpA_low_ee , temp_y)
            lp_filt_z = scipy.signal.filtfilt(flpB_low_ee , flpA_low_ee , temp_z)
            filt_sig_x = scipy.signal.filtfilt(fhp_eeB , fhp_eeA, lp_filt_x )
            filt_sig_y = scipy.signal.filtfilt(fhp_eeB , fhp_eeA, lp_filt_y )
            filt_sig_z = scipy.signal.filtfilt(fhp_eeB , fhp_eeA, lp_filt_z )
        
        if ee>=100 and ee<400:
            lp_filt_x = scipy.signal.filtfilt(flpB_med_ee , flpA_med_ee , temp_x)
            lp_filt_y = scipy.signal.filtfilt(flpB_med_ee , flpA_med_ee , temp_y)
            lp_filt_z = scipy.signal.filtfilt(flpB_med_ee , flpA_med_ee , temp_z)
            filt_sig_x = scipy.signal.filtfilt(fhp_eeB , fhp_eeA, lp_filt_x )
            filt_sig_y = scipy.signal.filtfilt(fhp_eeB , fhp_eeA, lp_filt_y )
            filt_sig_z = scipy.signal.filtfilt(fhp_eeB , fhp_eeA, lp_filt_z )
        
        if ee>=400:
            lp_filt_x = scipy.signal.filtfilt(flpB_high_ee , flpA_high_ee , temp_x)
            lp_filt_y = scipy.signal.filtfilt(flpB_high_ee , flpA_high_ee , temp_y)
            lp_filt_z = scipy.signal.filtfilt(flpB_high_ee , flpA_high_ee , temp_z)
            filt_sig_x = scipy.signal.filtfilt(fhpB_high_ee , fhpA_high_ee, lp_filt_x )
            filt_sig_y = scipy.signal.filtfilt(fhpB_high_ee , fhpA_high_ee, lp_filt_y )
            filt_sig_z = scipy.signal.filtfilt(fhpB_high_ee , fhpA_high_ee, lp_filt_z )
        
        pca_array = np.array([filt_sig_x , filt_sig_y ,filt_sig_z])
        pca_array = np.transpose(pca_array)
        pca = PCA(n_components= 3)
        pca.fit(pca_array)
        eig_x , eig_y , eig_z = pca.singular_values_
        wt_x = eig_x/(eig_x+eig_y+eig_z)
        wt_y = eig_y/(eig_x+eig_y+eig_z)
        wt_z = eig_z/(eig_x+eig_y+eig_z)
        final = (wt_x*filt_sig_x + wt_y*filt_sig_y + wt_z * filt_sig_z)
        lp_final = scipy.signal.filtfilt(fkern_lpB , fkern_lpA, final)
        hp_final = scipy.signal.filtfilt(fkern_hpB , fkern_hpA, lp_final)

        final_resp_sig[i] = np.append(final_resp_sig[i] , hp_final)
    #final_sig = np.array(final_resp_sig,dtype=object)
    
    return final_resp_sig
        #print(pca_array.shape)


# path = '/media/acrophase/pose1/charan/BR_Uncertainty/ACTUAL_BRUCE_DATA'
# srate = 256
# win_len = 32*srate

# # with open ("final_data","rb") as f:
# #    data = pkl.load(f)
# data = extract_data(path , srate , win_len)
# subject_id = 'S21'
# acc_x = data[subject_id]['ACC']['ACC_X'] 
# acc_y = data[subject_id]['ACC']['ACC_Y']
# acc_z = data[subject_id]['ACC']['ACC_Z']       
# resp = data[subject_id]['RESP']['RESP_DATA']

# fkern_lpB , fkern_lpA = low_pass(6,30,0.6)
# fkern_hpB , fkern_hpA = high_pass(4,20,0.1) 
    
# adr = resp_signal_extract(acc_x, acc_y , acc_z)
# ref_resp = []
# for i in range(len(resp)):
#     lp_filt = scipy.signal.filtfilt(fkern_lpB , fkern_lpA, resp[i])
#     ref_resp.append(scipy.signal.filtfilt(fkern_hpB , fkern_hpA, lp_filt))


# with open (subject_id+"_adr"+".pkl" , "wb") as f:
#     pkl.dump(adr , f)

# with open (subject_id+"_ref_resp"+".pkl" , "wb") as f:
#     pkl.dump(ref_resp , f)

# # print(type(ref_resp))
# # print(type(adr))
# # print(len(ref_resp))
# # print(len(adr))



