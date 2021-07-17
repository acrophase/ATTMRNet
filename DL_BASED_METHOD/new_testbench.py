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
import evidential_deep_learning as edl
import datetime
import os

srate = 700
win_length = 32*srate
lr = 1e-4
coeff_val = 1e-2
num_epochs = 5
model_input_shape = (128,3)

config = input("Enter the configuration")
data_path = 'C:/Users/ee19s/Desktop/Journal_Work/BR_Uncertainty/DL_BASED_METHOD/ppg_dalia_data'
data = extract_data(data_path , srate , win_length)
save_path = 'C:/Users/ee19s/Desktop/Journal_Work/BR_Uncertainty/DL_BASED_METHOD/SAVED_MODELS'
results_path = os.path.join(save_path , config.lower())
if not(os.path.isdir(results_path)):
   os.mkdir(results_path)  
#saved_model_path = os.path.join( 

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

annotation = pd.read_pickle('C:/Users/ee19s/Desktop/Journal_Work/BR_Uncertainty/DL_BASED_METHOD/annotation.pkl')
reference_rr = (annotation['Reference_RR'].values).reshape(-1,1)

tensor_input = tf.convert_to_tensor(input_data , dtype = 'float32')
tensor_output = tf.convert_to_tensor(output_data , dtype = 'float32')
tensor_ref_rr = tf.convert_to_tensor(reference_rr, dtype = 'float32')
training_ids = annotation['patient_id'] < 13

x_train_data = tensor_input[tf.convert_to_tensor(training_ids.values)]
x_test_data = tensor_input[tf.convert_to_tensor(~(training_ids.values))]
x_train_ref_rr = tensor_ref_rr[tf.convert_to_tensor(training_ids.values)]
x_test_ref_rr = tensor_ref_rr[tf.convert_to_tensor(~(training_ids.values))]

y_train_data = tensor_output[tf.convert_to_tensor(training_ids.values)]
y_test_data = tensor_output[tf.convert_to_tensor(~(training_ids.values))]

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/'+config.lower() + current_time + '/train'+'lr_'+str(lr)+"__"+'coeff_'+str(coeff_val)
test_log_dir = 'logs/gradient_tape/' +config.lower()+ current_time + '/test'+'lr_'+str(lr)+"__"+'coeff_'+str(coeff_val)
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

if config.lower() == "confc":
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_data , y_train_data))
    train_dataset = train_dataset.shuffle(len(x_train_data)).batch(128)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test_data , y_test_data))
    test_dataset = test_dataset.batch(128)

    model  = BRUnet(model_input_shape)
    optimizer = Adam(learning_rate = lr)
    #loss_fn = Huber()
    #loss_fn=edl.losses.EvidentialRegression
    for epoch in range(num_epochs):
        print("starting the epoch : {}".format(epoch + 1))
        train_loss_list = []
        for step, (x_batch_train , y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                #import pdb;pdb.set_trace()
                y_batch_train = tf.expand_dims(y_batch_train , axis = -1)
                output = model(x_batch_train , training = True)
                #mu, v, alpha, beta = tf.split(output, 4, axis=-1)
                loss_value = edl.losses.EvidentialRegression(y_batch_train,output,coeff = coeff_val)
                #loss_value = loss_fn(y_batch_train , output,coeff = 0.001)
                train_loss_list.append(loss_value)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights)) 
            train_loss(loss_value)
            # print(train_loss_list)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)

            if step%10 == 0:
                print('Epoch [%d/%d], lter [%d] Loss: %.4f'
                        %(epoch+1, num_epochs, step+1, loss_value))
        print("net loss -- {}".format(np.mean(np.array(train_loss_list))))
        test_loss_list = []
        best_loss = 100000

        for step , (x_batch_test,y_batch_test) in enumerate(test_dataset):
            y_batch_test = tf.expand_dims(y_batch_test , axis = -1)
            test_output = model(x_batch_test)
            test_loss_val = edl.losses.EvidentialRegression(y_batch_test , test_output , coeff = coeff_val)
            test_loss(test_loss_val)
            test_loss_list.append(test_loss_val)
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch)
        mean_loss = (sum(test_loss_list) / len(test_loss_list)) 
        if mean_loss < best_loss:
            best_loss = mean_loss
            model.save_weights(os.path.join(results_path, 'best_model'+'lr_'+str(lr)+'__'+'coeff_'+str(coeff_val)+'.h5'))
        print("validation loss -- {}".format(mean_loss))
        print(test_loss.result())
        train_loss.reset_states()
        test_loss.reset_states()

if config.lower() == "confd":
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_data , y_train_data, x_train_ref_rr))
    train_dataset = train_dataset.shuffle(len(x_train_data)).batch(128)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test_data , y_test_data, x_test_ref_rr))
    test_dataset = test_dataset.batch(128)

    model  = BRUnet_Multi_resp(model_input_shape)
    optimizer = Adam(learning_rate = lr)
    loss_fn = Huber()
    #loss_fn=edl.losses.EvidentialRegression
    for epoch in range(num_epochs):
        print("starting the epoch : {}".format(epoch + 1))
        train_loss_list = []
        for step, (x_batch_train , y_batch_train, x_batch_train_ref_rr) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                #import pdb;pdb.set_trace()
                y_batch_train = tf.expand_dims(y_batch_train , axis = -1)
                output, out_rr = model(x_batch_train , training = True)
                #mu, v, alpha, beta = tf.split(output, 4, axis=-1)
                #loss_value = edl.losses.EvidentialRegression(y_batch_train,output,coeff = coeff_val)
                loss_value = loss_fn(y_batch_train , output)
                loss_value_rr = loss_fn(x_batch_train_ref_rr, out_rr)
                net_loss_value = loss_value + loss_value_rr
                train_loss_list.append(net_loss_value)

            grads = tape.gradient(net_loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights)) 
            train_loss(net_loss_value)
            # print(train_loss_list)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)

            if step%10 == 0:
                print('Epoch [%d/%d], lter [%d] Loss: %.4f'
                        %(epoch+1, num_epochs, step+1, loss_value))
        print("net loss -- {}".format(np.mean(np.array(train_loss_list))))
        test_loss_list = []
        best_loss = 100000

        for step , (x_batch_test,y_batch_test,x_batch_test_ref_rr) in enumerate(test_dataset):
            y_batch_test = tf.expand_dims(y_batch_test , axis = -1)
            test_output,test_out_rr = model(x_batch_test)
            #test_loss_val = edl.losses.EvidentialRegression(y_batch_test , test_output , coeff = coeff_val)
            test_loss = loss_fn(y_batch_test , test_output)
            test_loss_rr = loss_fn(x_batch_test_ref_rr , test_out_rr)
            test_loss_val = test_loss + test_loss_rr
            test_loss(test_loss_val)
            test_loss_list.append(test_loss_val)
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch)
        mean_loss = (sum(test_loss_list) / len(test_loss_list)) 
        if mean_loss < best_loss:
            best_loss = mean_loss
            model.save_weights(os.path.join(results_path, 'best_model'+'lr_'+str(lr)+'__'+'coeff_'+str(coeff_val)+'.h5'))
        print("validation loss -- {}".format(mean_loss))
        print(test_loss.result())
        train_loss.reset_states()
        test_loss.reset_states()
