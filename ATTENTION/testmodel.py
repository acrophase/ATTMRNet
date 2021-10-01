import tensorflow as tf
from tf_model_new import BRUnet_raw

#import pdb;pdb.set_trace()
input_shape = (128, 2048, 3)
model_input_shape1 = (2048, 3)
#model_input_shape2 = (128, 3)
rand_input1 = tf.random.uniform(input_shape)
#rand_input2 = tf.random.uniform(input_shape)
model= BRUnet_raw(model_input_shape1)
output  = model(rand_input1)
print(output.shape)


