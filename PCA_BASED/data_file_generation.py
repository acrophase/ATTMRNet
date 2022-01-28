import pandas as pd
from data_extraction import *
from resp_signal_extraction import *
from rr_extration import *
import re
import pickle as pkl
import matplotlib.pyplot as plt
import sys
from scipy import signal
import argparse

fkern_lpB , fkern_lpA = low_pass(6,30,1)
fkern_hpB , fkern_hpA = high_pass(4,20,0.1)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="Path to data",default = '/media/acrophase/pose1/charan/BR_Uncertainty/ppg_dalia_data')
parser.add_argument("--srate", type=int, help="sampling rate", default=700)
parser.add_argument("--win_len", type=int, help="win length in secs", default=32)
args = parser.parse_args()

srate = args.srate
win_length = args.win_len * args.srate


data = extract_data(args.data_path, srate, win_length)

#with open("final_data", "wb") as f:
#    pkl.dump(data, f)

#with open ("final_data","rb") as f:
#    data = pkl.load(f)

for item in enumerate(data.keys()):
    ref_resp = []
    patient_id = item[1]
    
    acc_x = data[patient_id]["ACC"]["ACC_X"]
    acc_y = data[patient_id]["ACC"]["ACC_Y"]
    acc_z = data[patient_id]["ACC"]["ACC_Z"]
    resp = data[patient_id]["RESP"]["RESP_DATA"]
    activity_id = data[patient_id]["ACTIVITY_ID"]
    for i in range(len(resp)):
        lp_filt = scipy.signal.filtfilt(fkern_lpB , fkern_lpA, resp[i])
        ref_resp.append(scipy.signal.filtfilt(fkern_hpB , fkern_hpA, lp_filt))
    
    adr = resp_signal_extract(acc_x , acc_y , acc_z)
    print(adr.shape)
    print(type(ref_resp))
    #adr_new = adr_new.reshape(len(adr_new) , len(adr_new[0]))
    #avg_br_adr , _ = extremas_extraction(adr)
    #avg_br_ref , _ = extremas_extraction(ref_resp)

    #rr_adr  = (60*srate)/avg_br_adr
    #rr_ref  = (60*srate)/avg_br_ref
    #int_part = re.findall(r"\d+", patient_id)
    #sub_activity_ids = np.hstack((rr_adr.reshape(-1,1),rr_ref.reshape(-1, 1), np.array(activity_id).reshape(-1, 1),
    #                                    np.array([int(int_part[0])] * len(adr)).reshape(-1, 1),))  

    
    #adr = np.expand_dims(adr,1)
    #ref_resp = np.expand_dims(ref_resp,1)
    #import pdb;pdb.set_trace()
    #if item[0] == 0:
    #    final_windowed_inp = np.array(adr_new)  
    #    final_windowed_op = np.array(ref_resp_new)  
    #    #final_sub_activity_ids = sub_activity_ids
    #else:
    #    final_windowed_inp = np.vstack((final_windowed_inp, adr_new))
    #    final_windowed_op = np.vstack((final_windowed_op, ref_resp_new))
    #    #final_sub_activity_ids = np.vstack((final_sub_activity_ids, sub_activity_ids))
    #    #final_windowed_raw = np.vstack((final_windowed_raw, windowed_raw_sig))

#activity_df = pd.DataFrame(
#    final_sub_activity_ids, columns=["ADR_RR","Reference_RR", "activity_id", "patient_id"]
#)
#activity_df.to_pickle("annotation.pkl")
#with open("output", "wb") as f:
#    pkl.dump(final_windowed_op, f)

#with open("input", "wb") as f:
#    pkl.dump(final_windowed_inp, f)