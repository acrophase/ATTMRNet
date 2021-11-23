from student_model import BRUnet_Multi_resp_student
import tensorflow as tf

#import pdb;pdb.set_trace()
input_shape = (128, 128, 3)
model_input_shape = (128, 3)
rand_input = tf.random.uniform(input_shape)
model = BRUnet_Multi_resp_student(model_input_shape)
output,out_4  = model(rand_input)
print(output.shape)
print(out_4.shape)