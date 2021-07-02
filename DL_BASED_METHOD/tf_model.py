import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class IncBlock(tf.keras.Model):
    def __init__(self, in_channels, out_channels, size = 15, strides = 1):
        super(IncBlock, self).__init__()
        self.conv1x1 = layers.Conv1D(out_channels,kernel_size = 1,use_bias = False)

        self.conv1 = keras.Sequential([layers.Conv1D(out_channels//4 ,  kernel_size = size, strides = strides,  padding = 'same'
                                    ),
                                        layers.BatchNormalization(axis = 1)])

        self.conv2 = keras.Sequential([layers.Conv1D(out_channels//4 ,  kernel_size = 1, use_bias = False),
                                        layers.BatchNormalization(axis = 1),
                                        layers.LeakyReLU(alpha = 0.2),
                                        layers.Conv1D(out_channels//4 ,  kernel_size = size+2, strides = strides , padding = 'same'),
                                        layers.BatchNormalization(axis = 1)])
        
        self.conv3 = keras.Sequential([layers.Conv1D(out_channels//4 ,  kernel_size = 1, use_bias = False),
                                       layers.BatchNormalization(axis = 1),
                                       layers.LeakyReLU(alpha = 0.2),
                                       layers.Conv1D(out_channels//4, kernel_size = size + 4 , strides = strides,  padding = 'same'),
                                       layers.BatchNormalization(axis = 1)])

        self.conv4 = keras.Sequential([layers.Conv1D(out_channels//4 ,  kernel_size = 1, use_bias = False),
                                       layers.BatchNormalization(axis = 1),
                                       layers.LeakyReLU(alpha = 0.2),
                                       layers.Conv1D(out_channels//4, kernel_size = size + 6 , strides = strides,  padding = 'same'),
                                       layers.BatchNormalization(axis = 1)])
        
        self.relu = layers.ReLU()
    
    def call (self,x):
        res = self.conv1x1(x)
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)
        c4 = self.conv4(x)
        concat = layers.concatenate([c1,c2,c3,c4], axis = -1)
        concat+=res
        return self.relu(concat)
         
class BRUnet(tf.keras.Model):
    def __init__(self, in_channels):
        super(BRUnet,self).__init__()
        #in_channels = shape[1]
        
        self.en1 = keras.Sequential([layers.Conv1D(32,kernel_size = 3, padding = 'same',input_shape = in_channels),
                                    layers.BatchNormalization(axis = 1),
                                    layers.LeakyReLU(alpha = 0.2),
                                    layers.Conv1D(32, kernel_size = 5 , strides = 2 , padding = 'same')])

        self.en2 = keras.Sequential([layers.Conv1D(64, kernel_size = 3 , padding = 'same'),
                                     layers.BatchNormalization(axis = 1),
                                     layers.LeakyReLU(alpha = 0.2),
                                     layers.Conv1D(64 , kernel_size = 5,strides = 2, padding = 'same'),
                                     IncBlock(64,64)])

        self.en3 = keras.Sequential([layers.Conv1D(128, kernel_size = 3 , padding = 'same'),
                                     layers.BatchNormalization(axis = 1),
                                     layers.LeakyReLU(alpha = 0.2),
                                     layers.Conv1D(128 , kernel_size = 3,strides = 2, padding = 'same'),
                                     IncBlock(128,128)])

        self.en4 = keras.Sequential([layers.Conv1D(256, kernel_size = 3 , padding = 'same'),
                                     layers.BatchNormalization(axis = 1),
                                     layers.LeakyReLU(alpha = 0.2),
                                     layers.Conv1D(256 , kernel_size = 4,strides = 2,padding = 'same'),
                                     IncBlock(256,256)])

        self.en5 = keras.Sequential([layers.Conv1D(512, kernel_size = 3 , padding = 'same'),
                                     layers.BatchNormalization(axis = 1),
                                     layers.LeakyReLU(alpha = 0.2),
                                     layers.Conv1D(512 , kernel_size = 4,strides = 2,padding = 'same'),
                                     IncBlock(512,512)])

        self.en6 = keras.Sequential([layers.Conv1D(1024, kernel_size = 3 , padding = 'same'),
                                     layers.BatchNormalization(axis = 1),
                                     layers.LeakyReLU(alpha = 0.2),
                                     IncBlock(1024,1024)])
        
        self.de1_ecg = keras.Sequential([layers.Conv1DTranspose(512, kernel_size = 1,strides = 1),
                                         layers.BatchNormalization(axis = 1),
                                         layers.LeakyReLU(alpha = 0.2),
                                         IncBlock(512,512)])

        self.de2_ecg = keras.Sequential([layers.Conv1DTranspose(512, kernel_size = 1,strides = 2),
                                         layers.BatchNormalization(axis = 1),
                                         layers.LeakyReLU(alpha = 0.2),
                                         layers.Conv1DTranspose(256 , kernel_size = 1),
                                         IncBlock(256,256)])

        self.de3_ecg = keras.Sequential([layers.Conv1DTranspose(256, kernel_size = 1,strides = 1),  #kernel_size = 3
                                         layers.BatchNormalization(axis = 1),
                                         layers.LeakyReLU(alpha = 0.2),
                                         layers.Conv1DTranspose(128 , kernel_size = 1, strides = 2, padding = 'same'), #kernel_size = 4 strides = 2
                                         IncBlock(128,128)])

        self.de4_ecg = keras.Sequential([layers.Conv1DTranspose(64, kernel_size = 1,strides = 1), #kernel_size = 3
                                         layers.BatchNormalization(axis = 1),
                                         layers.LeakyReLU(alpha = 0.2),
                                         layers.Conv1DTranspose(64 , kernel_size = 1,strides = 2),  #kernel_size = 3 strides = 2
                                         IncBlock(64,64)])

        self.de5_ecg = keras.Sequential([layers.Conv1DTranspose(64, kernel_size = 1,strides = 1,padding = 'same'), #kernel_size = 3
                                         layers.BatchNormalization(axis = 1),   
                                         layers.LeakyReLU(alpha = 0.2),
                                         layers.Conv1DTranspose(32 , kernel_size = 1, strides = 2), #kernel_size = 3,strides = 2
                                         IncBlock(32,32)])

        self.de6_ecg = keras.Sequential([layers.Conv1DTranspose(32, kernel_size = 1, strides = 1), #kernel_size = 3
                                         layers.BatchNormalization(axis = 1),
                                         layers.LeakyReLU(alpha = 0.2),
                                         layers.Conv1DTranspose(16 , kernel_size = 1,strides = 2,padding = 'same'), #kernel_size = 3,strides = 2
                                         IncBlock(16,16)])
        
        self.de7_ecg = keras.Sequential([layers.Conv1DTranspose(1, kernel_size = 1,strides = 1,padding = 'same'),
                                         layers.LeakyReLU(alpha = 0.2)])

        self.de8_ecg = keras.Sequential([layers.Conv1DTranspose(1, kernel_size = 1,strides = 1,padding = 'same'),
                                         layers.LeakyReLU(alpha = 0.2)])
        
        self.de9_ecg = keras.Sequential([layers.Conv1DTranspose(1, kernel_size = 1,strides = 1,padding = 'same'),
                                         layers.LeakyReLU(alpha = 0.2)])
    
    def call (self,x):
        
        #import pdb;pdb.set_trace()
        e1 = self.en1(x)
        e2 = self.en2(e1)
        e3 = self.en3(e2)
        e4 = self.en4(e3)
        e5 = self.en5(e4)
        e6 = self.en6(e5)
        #import pdb;pdb.set_trace()
        d1_ecg = self.de1_ecg(e6)
        cat_ecg = layers.concatenate([d1_ecg , e5])
        d2_ecg = self.de2_ecg(cat_ecg)
        cat_ecg = layers.concatenate([d2_ecg , e4])
        d3_ecg = self.de3_ecg(cat_ecg)
        cat_ecg = layers.concatenate([d3_ecg , e3])
        
        d4_ecg = self.de4_ecg(cat_ecg)
        d4_ecg = d4_ecg[:,:,:-1]
        cat_ecg = layers.concatenate([d4_ecg , e2])
        d5_ecg = self.de5_ecg(cat_ecg)
        d5_ecg = d5_ecg[:,:,:-1]
        cat_ecg = layers.concatenate([d5_ecg , e1])
        d6_ecg = self.de6_ecg(cat_ecg)[:,:,:-1]
        d7_ecg = self.de7_ecg(d6_ecg)
        d8_ecg = self.de8_ecg(d7_ecg)
        d9_ecg = self.de9_ecg(d8_ecg)

        return d9_ecg




