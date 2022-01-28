import pickle as pkl
from rr_extration import *
import numpy as np

srate = 700
with open ("/media/acrophase/pose1/charan/BR_Uncertainty/BRUCE_PCA_BASED/signals/S17_adr.pkl","rb") as f:
    adr_17 = pkl.load(f)

with open ("/media/acrophase/pose1/charan/BR_Uncertainty/BRUCE_PCA_BASED/signals/S17_ref_resp.pkl","rb") as f:
    ref_resp_17 = pkl.load(f)

with open ("/media/acrophase/pose1/charan/BR_Uncertainty/BRUCE_PCA_BASED/signals/S18_adr.pkl","rb") as f:
    adr_18 = pkl.load(f)

with open ("/media/acrophase/pose1/charan/BR_Uncertainty/BRUCE_PCA_BASED/signals/S18_ref_resp.pkl","rb") as f:
    ref_resp_18 = pkl.load(f)

with open ("/media/acrophase/pose1/charan/BR_Uncertainty/BRUCE_PCA_BASED/signals/S19_adr.pkl","rb") as f:
    adr_19 = pkl.load(f)

with open ("/media/acrophase/pose1/charan/BR_Uncertainty/BRUCE_PCA_BASED/signals/S19_ref_resp.pkl","rb") as f:
    ref_resp_19 = pkl.load(f)

with open ("/media/acrophase/pose1/charan/BR_Uncertainty/BRUCE_PCA_BASED/signals/S20_adr.pkl","rb") as f:
    adr_20 = pkl.load(f)

with open ("/media/acrophase/pose1/charan/BR_Uncertainty/BRUCE_PCA_BASED/signals/S20_ref_resp.pkl","rb") as f:
    ref_resp_20 = pkl.load(f)

with open ("/media/acrophase/pose1/charan/BR_Uncertainty/BRUCE_PCA_BASED/signals/S21_adr.pkl","rb") as f:
    adr_21 = pkl.load(f)

with open ("/media/acrophase/pose1/charan/BR_Uncertainty/BRUCE_PCA_BASED/signals/S21_ref_resp.pkl","rb") as f:
    ref_resp_21 = pkl.load(f)

avg_dur_17,ext_17 = extremas_extraction(adr_17)
avg_dur_ref_17 , ext_ref_17 = extremas_extraction(ref_resp_17)

avg_dur_18,ext_18 = extremas_extraction(adr_18)
avg_dur_ref_18 , ext_ref_18 = extremas_extraction(ref_resp_18)

avg_dur_19,ext_19 = extremas_extraction(adr_19)
avg_dur_ref_19 , ext_ref_19 = extremas_extraction(ref_resp_19)

avg_dur_20,ext_20 = extremas_extraction(adr_20)
avg_dur_ref_20 , ext_ref_20 = extremas_extraction(ref_resp_20)

avg_dur_21,ext_21 = extremas_extraction(adr_21)
avg_dur_ref_21 , ext_ref_21 = extremas_extraction(ref_resp_21)

rr_17 = (60*srate)/avg_dur_17
rr_ref_17 = (60*srate)/avg_dur_ref_17

rr_18 = (60*srate)/avg_dur_18
rr_ref_18 = (60*srate)/avg_dur_ref_18

rr_19 = (60*srate)/avg_dur_19
rr_ref_19 = (60*srate)/avg_dur_ref_19

rr_20 = (60*srate)/avg_dur_20
rr_ref_20 = (60*srate)/avg_dur_ref_20

rr_21 = (60*srate)/avg_dur_21
rr_ref_21 = (60*srate)/avg_dur_ref_21

error_17= np.abs(rr_17 - rr_ref_17)
error_18 = np.abs(rr_18 - rr_ref_18)
error_19 = np.abs(rr_19 - rr_ref_19)
error_20 = np.abs(rr_20 - rr_ref_20)
error_21 = np.abs(rr_21 - rr_ref_21)

mae_17 = np.mean(error_17)
mae_18 = np.mean(error_18)
mae_19 = np.mean(error_19)
mae_20 = np.mean(error_20)
mae_21 = np.mean(error_21)

rmse_17 = np.sqrt(np.mean(error_17**2))
rmse_18 = np.sqrt(np.mean(error_18**2))
rmse_19 = np.sqrt(np.mean(error_19**2))
rmse_20 = np.sqrt(np.mean(error_20**2))
rmse_21 = np.sqrt(np.mean(error_21**2))

avg_mae = (mae_17+mae_18+mae_19+mae_20+mae_21)/5
avg_rmse = (rmse_17+rmse_18+rmse_19+rmse_20+rmse_21)/5

print(avg_mae)
print(avg_rmse)