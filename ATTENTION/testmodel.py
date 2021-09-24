import tensorflow as tf
from tf_model import BRUnet

#import pdb;pdb.set_trace()
input_shape = (128, 128, 3)
model_input_shape1 = (128, 3)
#model_input_shape2 = (128, 3)
rand_input1 = tf.random.uniform(input_shape)
#rand_input2 = tf.random.uniform(input_shape)
model = BRUnet(model_input_shape1)
output  = model(rand_input1)
print(output.shape)
