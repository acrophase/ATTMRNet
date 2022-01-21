import tensorflow as tf
import os
import numpy as np
import random
SEED = 0
#------------------------------------------------------------------------------------
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
#------------------------------------------------------------------------------------
def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

# Call the above function with seed value
set_global_determinism(seed=SEED)
#-----------------------------------------------------------------------------------

import pandas as pd
from data_extraction import *
from resp_signal_extraction import *
from rr_extration import *
from sklearn.preprocessing import MinMaxScaler
import re
import pickle as pkl
from bruce_student_model import *
from bruce_teacher_model import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import evidential_deep_learning as edl
import matplotlib.pyplot as plt
import datetime
import sys
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_model_path", type=str, help="Path to saved model", default='/media/acrophase/pose1/charan/BR_Uncertainty/BRUCE_KNOWLEDGE_DISTILL/SAVED_BRUCE_TEACH_KD_STU_MODEL'
)  #'/media/acrophase/pose1/Kapil/MultiRespDL/DL_Model/SAVED_MODELS')
parser.add_argument("--srate", type=int, help="sampling rate", default=256)
parser.add_argument("--win_len", type=int, help="win length in secs", default=32)
parser.add_argument("--num_epochs", type=int, help="number_of_epochs", default=100)
parser.add_argument(
    "--train_test_split_id", type=int, help="train test split id", default=17
)
parser.add_argument(
    "--annot_path", type=str, help="Path to annotation", default='/media/acrophase/pose1/charan/BR_Uncertainty/BRUCE_KNOWLEDGE_DISTILL/annotation.pkl'
)  #'/media/acrophase/pose1/Kapil/MultiRespDL/DL_Model/annotation.pkl')

args = parser.parse_args()

srate = args.srate
win_length = args.win_len * args.srate
num_epochs = args.num_epochs
train_test_split_id = args.train_test_split_id

with open("output", "rb") as f:
    output_data = pkl.load(f)

with open("input", "rb") as f:
    input_data = pkl.load(f)

with open("raw_signal.pkl", "rb") as f:
    raw_data = pkl.load(f)

input_data = np.transpose(input_data, (0, 2, 1))
raw_data = np.transpose(raw_data, (0, 2, 1))

input_data = np.around(input_data, decimals=4)
raw_data = np.around(raw_data, decimals=4)
output_data = np.around(output_data, decimals=4)

annotation = pd.read_pickle(
    args.annot_path
)  # pd.read_pickle('/media/acrophase/pose1/charan/MultiRespDL/DAYI_BIAN/annotation.pkl')

reference_rr = (annotation["Reference_RR"].values).reshape(-1, 1)
reference_rr = np.around(reference_rr, decimals=4)

tensor_input = tf.convert_to_tensor(input_data, dtype="float32")
tensor_output = tf.convert_to_tensor(output_data, dtype="float32")
tensor_ref_rr = tf.convert_to_tensor(reference_rr, dtype="float32")
tensor_raw_data = tf.convert_to_tensor(raw_data, dtype="float32")

training_ids = annotation["patient_id"] < train_test_split_id

x_train_data = tensor_input[tf.convert_to_tensor(training_ids.values)]
x_test_data = tensor_input[tf.convert_to_tensor(~(training_ids.values))]
x_train_ref_rr = tensor_ref_rr[tf.convert_to_tensor(training_ids.values)]
x_test_ref_rr = tensor_ref_rr[tf.convert_to_tensor(~(training_ids.values))]
x_train_raw_sig = tensor_raw_data[tf.convert_to_tensor(training_ids.values)]
x_test_raw_sig = tensor_raw_data[tf.convert_to_tensor(~(training_ids.values))]

y_train_data = tensor_output[tf.convert_to_tensor(training_ids.values)]
y_test_data = tensor_output[tf.convert_to_tensor(~(training_ids.values))]

def scheduler (epoch):
    if epoch <=20:
        lr = 1e-2
    else:
        lr = 1e-3
    return lr
model_input_shape = (128,3)
model_student  = BRUnet_Multi_resp_student_2(model_input_shape)
model_teacher = BRUnet_Multi_resp_ATT_MC(model_input_shape)
loss_fn = Huber()
save_path = (args.save_model_path)
results_path = os.path.join(save_path , "TEAC_KD_STU_MODEL")
if not(os.path.isdir(results_path)):
    os.mkdir(results_path)        
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'evi_teac_kd_stu/logs/gradient_tape/'+ current_time + '/train'
test_log_dir = 'evi_teac_kd_stu/logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_data , y_train_data, x_train_ref_rr))
train_dataset = train_dataset.shuffle(len(x_train_data)).batch(128)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test_data , y_test_data, x_test_ref_rr))
test_dataset = test_dataset.batch(128)

print("Starting the training")
for epoch in tqdm.tqdm(range(num_epochs)):
    print("starting the epoch : {}".format(epoch + 1))
    train_loss_list = []
    optimizer = Adam(learning_rate = scheduler(epoch))
    for step, (x_batch_train , y_batch_train, x_batch_train_ref_rr) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            out_st,out_rr_st,e6_st,at1_st,at2_st,at4_st,at5_st,at6_st,at7_st= model_student(x_batch_train , training = True)
            out_tea, out_rr_tea,e6_tea,at1_tea,at2_tea,at3_tea,at4_tea,at5_tea,\
            at6_tea,at7_tea,at8_tea,at9_tea = model_teacher(x_batch_train , training = True)
            loss_1 = loss_fn(out_tea , out_st)
            loss_2 = loss_fn(out_rr_tea , out_rr_st)
            loss_3 = loss_fn(e6_tea,e6_st)
            loss_4 = loss_fn(at1_tea,at1_st)
            loss_5 = loss_fn(at4_tea, at4_st)
            #loss_value_rr = loss_fn(x_batch_train_ref_rr, out_rr)
            #net_loss_value = loss_value + loss_value_rr
            net_loss_value = loss_1+loss_2+loss_3+loss_4+loss_5
            train_loss_list.append(net_loss_value)  
        grads = tape.gradient(net_loss_value, model_student.trainable_weights)
        optimizer.apply_gradients(zip(grads, model_student.trainable_weights)) 
        train_loss(net_loss_value)
        #print(out_rr)
        #print("###############################################")
        #print(out_rr)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)

        if step%10 == 0:
            print('Epoch [%d/%d], lter [%d] Loss: %.4f'
                    %(epoch+1, num_epochs, step+1, net_loss_value))
    print("net loss -- {}".format(np.mean(np.array(train_loss_list))))
    test_loss_list = []
    best_loss = 100000

    for step , (x_batch_test,y_batch_test,x_batch_test_ref_rr) in enumerate(test_dataset):
        test_out_st,test_out_rr_st,test_e6_st,test_at1_st,test_at2_st,test_at4_st,test_at5_st,\
                                               test_at6_st,test_at7_st= model_student(x_batch_test , training = True)
        
        test_out_tea, test_out_rr_tea,test_e6_tea,test_at1_tea,test_at2_tea,test_at3_tea,test_at4_tea,test_at5_tea,\
        test_at6_tea,test_at7_tea,test_at8_tea,test_at9_tea = model_teacher(x_batch_test , training = True)
        #test_loss_resp = loss_fn(y_batch_test , test_output)
        #test_loss_rr = loss_fn(x_batch_test_ref_rr , test_out_rr)
        test_loss_1 = loss_fn(test_out_st , test_out_tea)
        test_loss_2 = loss_fn(test_out_rr_tea , test_out_rr_st)
        test_loss_3 = loss_fn(test_e6_tea , test_e6_st)
        test_loss_4 = loss_fn(test_at1_tea , test_at1_st)
        test_loss_5 = loss_fn(test_at4_tea , test_at4_st)
        #test_loss_val = test_loss_resp + test_loss_rr
        test_loss_val = test_loss_1 + test_loss_2 + test_loss_3 + test_loss_4 + test_loss_5
        test_loss(test_loss_val)
        test_loss_list.append(test_loss_val)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
        #print(test_out_rr)
    mean_loss = (sum(test_loss_list) / len(test_loss_list)) 
    if mean_loss < best_loss:
        best_loss = mean_loss
            #model.save_weights(os.path.join(results_path, 'best_model_1'+str(1e-3)+'_'+str(num_epochs)+'.h5'))
        model_student.save_weights(os.path.join(results_path, 'best_model_5'+str(1e-2)+'_'+str(1e-3)+'_'+str(num_epochs)+'.h5'))
    print("validation loss -- {}".format(mean_loss))
    print(test_loss.result())
    train_loss.reset_states()
    test_loss.reset_states()
