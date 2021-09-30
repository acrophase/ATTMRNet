import tensorflow as tf
from tf_model import BRUnet_raw_multi

#import pdb;pdb.set_trace()
input_shape = (128, 2048, 3)
model_input_shape1 = (2048, 3)
#model_input_shape2 = (128, 3)
rand_input1 = tf.random.uniform(input_shape)
#rand_input2 = tf.random.uniform(input_shape)
model= BRUnet_raw_multi(model_input_shape1)
output,out_4,attn1,attn2,attn3,attn4,attn5,attn6,attn7,attn8,attn9,attn10,attn11  = model(rand_input1)
print(output.shape)
print(out_4.shape)

