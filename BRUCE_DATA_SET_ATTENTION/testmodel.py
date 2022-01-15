from tf_model_attn_monte import BRUnet_Multi_resp_ATT_MC
import tensorflow as tf

#import pdb;pdb.set_trace()
input_shape = (128, 128, 3)
model_input_shape = (128, 3)
rand_input = tf.random.uniform(input_shape)
model = BRUnet_Multi_resp_ATT_MC(model_input_shape)
output,out_4,attn1,attn2,attn3,attn4,attn5,attn6,attn7,attn8,attn9  = model(rand_input)
print(output.shape)
print(out_4.shape)