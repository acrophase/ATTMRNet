import tensorflow as tf
from tf_model import BRUnet_raw_encoder

#import pdb;pdb.set_trace()
input_shape = (128, 2048, 3)
model_input_shape1 = (2048, 3)
#model_input_shape2 = (128, 3)
rand_input1 = tf.random.uniform(input_shape)
#rand_input2 = tf.random.uniform(input_shape)
model= BRUnet_raw_encoder(model_input_shape1)
output,_,_,_,_  = model(rand_input1)
print(output.shape)
