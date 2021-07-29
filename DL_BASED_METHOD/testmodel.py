from tf_model import BRUnet,BRUnet_Multi_resp,BRUnet_Encoder,BRUnet_raw,BRUnet_raw_encoder,BRUnet_raw_multi
import tensorflow as tf

#import pdb;pdb.set_trace()
input_shape = (128, 2048, 3)
model_input_shape = (2048, 3)
rand_input = tf.random.uniform(input_shape)
model = BRUnet_raw_multi(model_input_shape)
output,out_4  = model(rand_input)
print(output.shape)
print(out_4.shape)