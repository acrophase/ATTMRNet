import pickle as pkl
from rr_extration import *
import numpy as np

srate = 700
with open ("/media/acrophase/pose1/charan/BR_Uncertainty/PCA_BASED/S13_signals/S13_adr.pkl","rb") as f:
    adr_13 = pkl.load(f)

with open ("/media/acrophase/pose1/charan/BR_Uncertainty/PCA_BASED/S13_signals/S13_ref_resp.pkl","rb") as f:
    ref_resp_13 = pkl.load(f)

with open ("/media/acrophase/pose1/charan/BR_Uncertainty/PCA_BASED/S14_signals/S14_adr.pkl","rb") as f:
    adr_14 = pkl.load(f)

with open ("/media/acrophase/pose1/charan/BR_Uncertainty/PCA_BASED/S14_signals/S14_ref_resp.pkl","rb") as f:
    ref_resp_14 = pkl.load(f)

with open ("/media/acrophase/pose1/charan/BR_Uncertainty/PCA_BASED/S15_signals/S15_adr.pkl","rb") as f:
    adr_15 = pkl.load(f)

with open ("/media/acrophase/pose1/charan/BR_Uncertainty/PCA_BASED/S15_signals/S15_ref_resp.pkl","rb") as f:
    ref_resp_15 = pkl.load(f)

avg_dur_13,ext_13 = extremas_extraction(adr_13)
avg_dur_ref_13 , ext_ref_13 = extremas_extraction(ref_resp_13)

avg_dur_14,ext_14 = extremas_extraction(adr_14)
avg_dur_ref_14 , ext_ref_14 = extremas_extraction(ref_resp_14)

avg_dur_15,ext_15 = extremas_extraction(adr_15)
avg_dur_ref_15 , ext_ref_15 = extremas_extraction(ref_resp_15)

rr_13 = (60*srate)/avg_dur_13
rr_ref_13 = (60*srate)/avg_dur_ref_13

rr_14 = (60*srate)/avg_dur_14
rr_ref_14 = (60*srate)/avg_dur_ref_14

rr_15 = (60*srate)/avg_dur_15
rr_ref_15 = (60*srate)/avg_dur_ref_15

error_13 = np.abs(rr_13 - rr_ref_13)
error_14 = np.abs(rr_14 - rr_ref_14)
error_15 = np.abs(rr_15 - rr_ref_15)

mae_13 = np.mean(error_13)
mae_14 = np.mean(error_14)
mae_15 = np.mean(error_15)
rmse_13 = np.sqrt(np.mean(error_13**2))
rmse_14 = np.sqrt(np.mean(error_14**2))
rmse_15 = np.sqrt(np.mean(error_15**2))

avg_mae = (mae_13+mae_14+mae_15)/3
avg_rmse = (rmse_13+rmse_14+rmse_15)/3

print(avg_mae)
print(avg_rmse)