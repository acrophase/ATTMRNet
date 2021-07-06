import numpy as np
import pandas as pd
from data_extraction import *
from resp_signal_extraction import *
from rr_extration import *
from sklearn.preprocessing import MinMaxScaler
import re
import tensorflow as tf
import pickle as pkl
from tf_model import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

srate = 700
win_length = 32*srate

data_path = '/media/hticpose/drive1/charan/BR_Uncertainty/ppg_dalia_data'
data = extract_data(data_path , srate , win_length)

for item in enumerate(data.keys()):
    patient_id = item[1]  
    ecg = data[patient_id]['ECG']['ECG_DATA']
    rpeaks = data[patient_id]['ECG']['RPEAKS']
    amps = data[patient_id]['ECG']['AMPLITUDES']
    acc = data[patient_id]['ACC']['ACC_DATA']
    resp = data[patient_id]['RESP']['RESP_DATA']
    activity_id = data[patient_id]['ACTIVITY_ID']
    scaler = MinMaxScaler()

    edr_hrv , edr_rpeak , adr , ref_resp = edr_adr_extraction(acc, rpeaks , amps , resp)

    for i in range(len(edr_hrv)):
        edr_hrv[i] = np.append(edr_hrv[i] , np.zeros(128 - len(edr_hrv[i])))
        edr_rpeak[i] = np.append(edr_rpeak[i] , np.zeros(128 - len(edr_rpeak[i])))
        adr[i] = np.append(adr[i] , np.zeros(128 - len(adr[i])))
        ref_resp[i] = np.append(ref_resp[i] , np.zeros(128 - len(ref_resp[i])))
    ref_rr_duration, _ =  extremas_extraction(ref_resp)
    ref_rr = (60*4)/ref_rr_duration

    edr_hrv , edr_rpeak , adr , ref_resp = np.expand_dims(np.asarray(edr_hrv), axis = -1), np.expand_dims(np.asarray(edr_rpeak), axis = -1)\
                               , np.expand_dims(np.asarray(adr), axis =-1) , np.expand_dims(np.asarray(ref_resp), axis =-1)
    
    edr_hrv = scaler.fit_transform(edr_hrv.reshape(len(edr_hrv),len(edr_hrv[0])))
    edr_rpeak = scaler.fit_transform(edr_rpeak.reshape(len(edr_rpeak),len(edr_rpeak[0])))
    adr = scaler.fit_transform(adr.reshape(len(adr),len(adr[0])))
    ref_resp = scaler.fit_transform(ref_resp.reshape(len(ref_resp),len(ref_resp[0])))

    windowed_inp = np.concatenate((np.expand_dims(edr_hrv, 1), np.expand_dims(edr_rpeak, 1), np.expand_dims(adr, 1)), axis = 1)
    int_part  = re.findall(r'\d+', patient_id)

    sub_activity_ids = np.hstack((ref_rr.reshape(-1,1),np.array(activity_id).reshape(-1,1), np.array([int(int_part[0])]*len(edr_hrv)).reshape(-1,1)))
    
    if item[0] == 0:
        final_windowed_inp = windowed_inp
        final_windowed_op = np.array(ref_resp)
        final_sub_activity_ids = sub_activity_ids
    else:
        final_windowed_inp = np.vstack((final_windowed_inp , windowed_inp))
        final_windowed_op = np.vstack((final_windowed_op , ref_resp))
        final_sub_activity_ids = np.vstack((final_sub_activity_ids , sub_activity_ids))

with open('output','rb') as f:
    output_data = pkl.load(f)

with open('input','rb') as f:
    input_data = pkl.load(f)

input_data = input_data.reshape(input_data.shape[0],input_data.shape[-1],input_data.shape[1])

annotation = pd.read_pickle('/media/hticpose/drive1/charan/BR_Uncertainty/DL_BASED_METHOD/annotation.pkl')
reference_rr = (annotation['Reference_RR'].values).reshape(-1,1)

tensor_input = tf.convert_to_tensor(input_data)
tensor_output = tf.convert_to_tensor(output_data)
tensor_ref_rr = tf.convert_to_tensor(reference_rr)
training_ids = annotation['patient_id'] < 13

x_train_data = tensor_input[tf.convert_to_tensor(training_ids.values)]
x_test_data = tensor_input[tf.convert_to_tensor(~(training_ids.values))]
x_train_ref_rr = tensor_ref_rr[tf.convert_to_tensor(training_ids.values)]
x_test_ref_rr = tensor_ref_rr[tf.convert_to_tensor(~(training_ids.values))]

y_train_data = tensor_output[tf.convert_to_tensor(training_ids.values)]
y_test_data = tensor_output[tf.convert_to_tensor(~(training_ids.values))]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train_data , y_train_data))
train_dataset = train_dataset.shuffle(len(x_train_data)).batch(128)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test_data , y_test_data))
test_dataset = test_dataset.batch(128)

model_input_shape = (128,3)

model  = BRUnet(model_input_shape)
optimizer = Adam(learning_rate = 0.005)
loss_fn = Huber()
num_epochs = 1000

for epoch in range(num_epochs):
    print("starting the epoch : {}".format(epoch + 1))
    train_loss_list = []
    for step, (x_batch_train , y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            output = model(x_batch_train , training = True)
            loss_value = loss_fn(y_batch_train , output)
            train_loss_list.append(loss_value)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights)) 

        if step%10 == 0:
            print('Epoch [%d/%d], lter [%d] Loss: %.4f'
                    %(epoch+1, num_epochs, step+1, loss_value))
    print("net loss -- {}".format(np.mean(np.array(train_loss_list))))
    test_loss_list = []
    best_loss = 10000

    for step , (x_batch_test,y_batch_test) in enumerate(test_dataset):
        test_output = model(x_batch_test)
        test_loss = loss_fn(y_batch_test , test_output)
        test_loss_list.append(test_loss)
    mean_loss = (sum(test_loss_list) / len(test_loss_list)) 
    if mean_loss < best_loss:
        best_loss = mean_loss
        model.save_weights('/media/hticpose/drive1/charan/BR_Uncertainty/DL_BASED_METHOD/best_model.h5')
    print("validation loss -- {}".format(mean_loss))

model.summary()
