from tf_model import BRUnet,BRUnet_Multi_resp,BRUnet_Encoder
import tensorflow as tf

#import pdb;pdb.set_trace()
input_shape = (128, 128, 3)
model_input_shape = (128, 3)
rand_input = tf.random.uniform(input_shape)
model = BRUnet_Multi_resp(model_input_shape)
output,out_4 = model(rand_input)
print(out_4.shape)
