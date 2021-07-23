from tf_model import BRUnet,BRUnet_Multi_resp,BRUnet_Encoder,BRUnet_raw
import tensorflow as tf

#import pdb;pdb.set_trace()
input_shape = (128, 2048, 3)
model_input_shape = (2048, 3)
rand_input = tf.random.uniform(input_shape)
model = BRUnet_raw(model_input_shape)
output = model(rand_input)
print(output)
