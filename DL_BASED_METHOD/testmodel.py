from tf_model import BRUnet
import tensorflow as tf

#import pdb;pdb.set_trace()
input_shape = (128, 128, 3)
model_input_shape = (128, 3)
rand_input = tf.random.uniform(input_shape)
model = BRUnet(model_input_shape)
output = model(rand_input)
print(output)
