import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import evidential_deep_learning as edl
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

#class Conv1DTranspose(tf.keras.Model):
#    def __init__(self, filters, kernel_size, strides=2, padding='same'):
#        """
#            input_tensor: tensor, with the shape (batch_size, time_steps, dims)
#            filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
#            kernel_size: int, size of the convolution kernel
#            strides: int, convolution step size
#            padding: 'same' | 'valid'
#        """
#        super(Conv1DTranspose, self).__init__()
#        self.obj = keras.Sequential([  layers.Lambda(lambda x: tf.expand_dims(x, axis=2)),
#                            layers.Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding
#                            ,kernel_regularizer=l2(lam),bias_regularizer=l2(lam)),
#                            layers.Lambda(lambda x: tf.squeeze(x, axis=2))])
#
#    def call (self,x):
#        return self.obj(x)

class AttentionBlock(tf.keras.Model):
    def __init__(self,*args):
        super(AttentionBlock,self).__init__()
        if len(args) == 2:
            g_shape = args[0]
            x_shape = args[1]
            self.W_g = keras.Sequential([layers.Conv1D(g_shape,kernel_size = 1,strides = 1,padding = 'same'),
                                       layers.BatchNormalization(axis = -1)])

            self.W_x = keras.Sequential([layers.Conv1D(x_shape,kernel_size = 1,strides = 1,padding = 'same'),
                                       layers.BatchNormalization(axis = -1)])

            self.psi = keras.Sequential([layers.Conv1D(1,kernel_size = 1,strides = 1,padding = 'same'),
                                        layers.BatchNormalization(axis = -1),
                                        layers.Activation(activation='sigmoid')])

            self.relu = layers.ReLU()
        
        elif len(args) == 1:
            x_shape = args[0]
            self.W_x = keras.Sequential([layers.Conv1D(x_shape,kernel_size = 1,strides = 1,padding = 'same'),
                                       layers.BatchNormalization(axis = -1)])

            self.psi = keras.Sequential([layers.Conv1D(1,kernel_size = 1,strides = 1,padding = 'same'),
                                        layers.BatchNormalization(axis = -1),
                                        layers.Activation(activation='sigmoid')])

            self.relu = layers.ReLU()
    
    def call(self,*args):
        if len(args) == 2:
            #import pdb;pdb.set_trace()
            g = args[0]
            x = args[1]
            g1 = self.W_g(g)
            x1 = self.W_x(x)
            psi = self.relu(g1 + x1)
            psi = self.psi(psi)
            out = x * psi
            return out

        else:
            #import pdb;pdb.set_trace()
            x = args[0]
            x1 = self.W_x(x)
            psi = self.relu(x1)
            psi = self.psi(psi)
            out = x * psi
            return out


#class AttentionBlock(tf.keras.Model):
#    def __init__(self,g_shape,x_shape):
#        super(AttentionBlock , self).__init__()

#        self.W_g = keras.Sequential([layers.Conv1D(g_shape,kernel_size = 1,strides = 1,padding = 'same'),
#                                       layers.BatchNormalization(axis = -1)])

#        self.W_x = keras.Sequential([layers.Conv1D(x_shape,kernel_size = 1,strides = 1,padding = 'same'),
#                                       layers.BatchNormalization(axis = -1)])

#        self.psi = keras.Sequential([layers.Conv1D(1,kernel_size = 1,strides = 1,padding = 'same'),
#                                        layers.BatchNormalization(axis = -1),
#                                        layers.Activation(activation='sigmoid')])

#        self.relu = layers.ReLU()
    
#    def call (self,g,x):
        #import pdb;pdb.set_trace()
#        g1 = self.W_g(g)
#        x1 = self.W_x(x)
#        psi = self.relu(g1 + x1)
#        psi = self.psi(psi)
#        out = x * psi
#        return out

#class AttentionBlock_new(tf.keras.Model):
#    def __init__(self,x_shape):
#        super(AttentionBlock_new , self).__init__()

#        self.W_x = keras.Sequential([layers.Conv1D(x_shape,kernel_size = 1,strides = 1,padding = 'same'),
#                                       layers.BatchNormalization(axis = -1)])

#        self.psi = keras.Sequential([layers.Conv1D(1,kernel_size = 1,strides = 1,padding = 'same'),
#                                        layers.BatchNormalization(axis = -1),
#                                        layers.Activation(activation='sigmoid')])

#        self.relu = layers.ReLU()
    
#    def call (self,x):
#        #import pdb;pdb.set_trace()
#        x1 = self.W_x(x)
#        psi = self.relu(x1)
#        psi = self.psi(psi)
#        out = x * psi
#        return out

class IncBlock(tf.keras.Model):
    def __init__(self, in_channels, out_channels, size = 15, strides = 1):
        super(IncBlock, self).__init__()
        self.conv1x1 = layers.Conv1D(out_channels,kernel_size = 1,use_bias = False)

        self.conv1 = keras.Sequential([layers.Conv1D(out_channels//4 ,  kernel_size = size, strides = strides,  padding = 'same'),
                                        layers.BatchNormalization(axis = -1)])

        self.conv2 = keras.Sequential([layers.Conv1D(out_channels//4 ,  kernel_size = 1, use_bias = False),
                                        layers.BatchNormalization(axis = -1),
                                        layers.LeakyReLU(alpha = 0.2),
                                        layers.Conv1D(out_channels//4 ,  kernel_size = size+2, strides = strides , padding = 'same'),
                                        layers.BatchNormalization(axis = -1)])
        
        self.conv3 = keras.Sequential([layers.Conv1D(out_channels//4 ,  kernel_size = 1, use_bias = False),
                                       layers.BatchNormalization(axis = -1),
                                       layers.LeakyReLU(alpha = 0.2),
                                       layers.Conv1D(out_channels//4, kernel_size = size + 4 , strides = strides,  padding = 'same'),
                                       layers.BatchNormalization(axis = -1)])

        self.conv4 = keras.Sequential([layers.Conv1D(out_channels//4 ,  kernel_size = 1, use_bias = False),
                                       layers.BatchNormalization(axis = -1),
                                       layers.LeakyReLU(alpha = 0.2),
                                       layers.Conv1D(out_channels//4, kernel_size = size + 6 , strides = strides,  padding = 'same'),
                                       layers.BatchNormalization(axis = -1)])
        
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

class Subsampling(tf.keras.Model):
    def __init__(self,in_channels):
        super(Subsampling,self).__init__()
        
        self.layer1 = keras.Sequential([layers.Conv1D(3 , kernel_size = 4 , padding = 'same',input_shape = in_channels),
                                        layers.BatchNormalization(axis = 1),
                                        layers.LeakyReLU(alpha = 0.2),
                                        layers.MaxPooling1D(pool_size=2, strides=None, padding="same"),
                                        IncBlock(3,128)])
        
        self.layer2 = keras.Sequential([layers.Conv1D(128 , kernel_size = 4 , padding = 'same'),
                                        layers.BatchNormalization(axis = 1),
                                        layers.LeakyReLU(alpha = 0.2),
                                        layers.MaxPooling1D(pool_size=2, strides=None, padding="same"),
                                        IncBlock(128,256)])
        
        self.layer3 = keras.Sequential([layers.Conv1D(256 , kernel_size = 4 , padding = 'same'),
                                        layers.BatchNormalization(axis = 1),
                                        layers.LeakyReLU(alpha = 0.2),
                                        layers.MaxPooling1D(pool_size=2, strides=None, padding="same"),
                                        IncBlock(256,512)])
        
        self.layer4 = keras.Sequential([layers.Conv1D(512 , kernel_size = 4 , padding = 'same'),
                                        layers.BatchNormalization(axis = 1),
                                        layers.LeakyReLU(alpha = 0.2),
                                        layers.MaxPooling1D(pool_size=2, strides=None, padding="same"),
                                        IncBlock(512,512)])
        self.layer5 = layers.Conv1D(3 , kernel_size = 4 , padding = 'same')

    def call(self,x):
        #import pdb;pdb.set_trace()
        out_1 = self.layer1(x)
        out_2 = self.layer2(out_1)
        out_3 = self.layer3(out_2)
        out_4 = self.layer4(out_3)
        out_5 = self.layer5(out_4)
        return out_5
         
class BRUnet(tf.keras.Model):
    def __init__(self, in_channels):
        super(BRUnet,self).__init__()
        #in_channels = shape[1]
        
        self.en1 = keras.Sequential([layers.Conv1D(32,kernel_size = 3, padding = 'same',input_shape = in_channels),
                                    layers.BatchNormalization(axis = -1),
                                    layers.LeakyReLU(alpha = 0.2),
                                    layers.Conv1D(32, kernel_size = 5 , strides = 2 , padding = 'same'),
                                    IncBlock(32,32)])

        self.en2 = keras.Sequential([layers.Conv1D(64, kernel_size = 3 , padding = 'same'),
                                     layers.BatchNormalization(axis = -1),
                                     layers.LeakyReLU(alpha = 0.2),
                                     layers.Conv1D(64 , kernel_size = 5,strides = 2, padding = 'same'),
                                     IncBlock(64,64)])

        self.en3 = keras.Sequential([layers.Conv1D(128, kernel_size = 3 , padding = 'same'),
                                     layers.BatchNormalization(axis = -1),
                                     layers.LeakyReLU(alpha = 0.2),
                                     layers.Conv1D(128 , kernel_size = 3,strides = 2, padding = 'same'),
                                     IncBlock(128,128)])

        self.en4 = keras.Sequential([layers.Conv1D(256, kernel_size = 3 , padding = 'same'),
                                     layers.BatchNormalization(axis = -1),
                                     layers.LeakyReLU(alpha = 0.2),
                                     layers.Conv1D(256 , kernel_size = 4,strides = 2,padding = 'same'),
                                     IncBlock(256,256)])

        self.en5 = keras.Sequential([layers.Conv1D(512, kernel_size = 3 , padding = 'same'),
                                     layers.BatchNormalization(axis = -1),
                                     layers.LeakyReLU(alpha = 0.2),
                                     layers.Conv1D(512 , kernel_size = 4,strides = 2,padding = 'same'),
                                     IncBlock(512,512)])

        self.en6 = keras.Sequential([layers.Conv1D(1024, kernel_size = 3 , padding = 'same'),
                                     layers.BatchNormalization(axis = -1),
                                     layers.LeakyReLU(alpha = 0.2),
                                     IncBlock(1024,1024)])
        
        self.de1_ecg = keras.Sequential([layers.Conv1DTranspose(512, kernel_size = 1,strides = 1 , padding = 'same'),
                                         layers.BatchNormalization(axis = -1),
                                         layers.LeakyReLU(alpha = 0.2),
                                         IncBlock(512,512)])
        #self.attn1 = AttentionBlock(512,512)

        self.de2_ecg = keras.Sequential([layers.Conv1DTranspose(512, kernel_size = 1,strides = 2,padding = 'same'),
                                         layers.BatchNormalization(axis = -1),
                                         layers.LeakyReLU(alpha = 0.2),
                                         layers.Conv1DTranspose(256 , kernel_size = 1,strides = 1,padding = 'same'),
                                         IncBlock(256,256)])
        
        #self.attn2 = AttentionBlock(256, 256)

        self.de3_ecg = keras.Sequential([layers.Conv1D(256, kernel_size = 1,strides = 1),  #kernel_size = 3
                                         layers.BatchNormalization(axis = -1),
                                         layers.LeakyReLU(alpha = 0.2),
                                         layers.Conv1DTranspose(128 , kernel_size = 4, strides = 2, padding = 'same'), #kernel_size = 4 strides = 2
                                         IncBlock(128,128)])
        
        #self.attn3 = AttentionBlock(128, 128)

        self.de4_ecg = keras.Sequential([layers.Conv1D(128, kernel_size = 1,strides = 1), #kernel_size = 3
                                         layers.BatchNormalization(axis = -1),
                                         layers.LeakyReLU(alpha = 0.2),
                                         layers.Conv1DTranspose(64 , kernel_size = 3,strides = 2,padding = 'same'),  #kernel_size = 3 strides = 2
                                         IncBlock(64,64)])
        
        #self.attn4 = AttentionBlock(64, 64)

        self.de5_ecg = keras.Sequential([layers.Conv1D(64, kernel_size = 1,strides = 1,padding = 'same'), #kernel_size = 3
                                         layers.BatchNormalization(axis = -1),   
                                         layers.LeakyReLU(alpha = 0.2),
                                         layers.Conv1DTranspose(32 , kernel_size = 3, strides = 2,padding = 'same'), #kernel_size = 3,strides = 2
                                         IncBlock(32,32)])
        
        #self.attn5 = AttentionBlock(32, 32)

        self.de6_ecg = keras.Sequential([layers.Conv1D(32, kernel_size = 1, strides = 1), #kernel_size = 3
                                         layers.BatchNormalization(axis = -1),
                                         layers.LeakyReLU(alpha = 0.2),
                                         layers.Conv1DTranspose(16 , kernel_size = 3,strides = 2,padding = 'same'), #kernel_size = 3,strides = 2
                                         IncBlock(16,16)])
        
        self.de7_ecg = keras.Sequential([layers.Conv1DTranspose(1, kernel_size = 1,strides = 1,padding = 'same'),
                                         layers.LeakyReLU(alpha = 0.2)])

        self.de8_ecg = keras.Sequential([layers.Conv1DTranspose(1, kernel_size = 1,strides = 1,padding = 'same'),
                                         layers.LeakyReLU(alpha = 0.2)])
        
        self.de9_ecg = keras.Sequential([layers.Conv1DTranspose(1, kernel_size = 1,strides = 1,padding = 'same'),
                                         layers.LeakyReLU(alpha = 0.2)])
        
        #self.ev1 = edl.layers.DenseNormalGamma(1)
    
    def call (self,x,training = False):
        #import pdb;pdb.set_trace()
        e1 = self.en1(x)
        e2 = self.en2(e1)
        e3 = self.en3(e2)
        e4 = self.en4(e3)
        e5 = self.en5(e4)
        e6 = self.en6(e5)
        
        d1_ecg = self.de1_ecg(e6)
        #attn1 = self.attn1(d1_ecg , e5)
        cat_ecg = layers.concatenate([d1_ecg , e5])
        d2_ecg = self.de2_ecg(cat_ecg)
        #attn2 = self.attn2(d2_ecg , e4)
        cat_ecg = layers.concatenate([d2_ecg , e4])
        d3_ecg = self.de3_ecg(cat_ecg)
        #attn3 = self.attn3(d3_ecg , e3)
        cat_ecg = layers.concatenate([d3_ecg , e3])

        #import pdb;pdb.set_trace()
        d4_ecg = self.de4_ecg(cat_ecg)
        d4_ecg = d4_ecg[:,:,:-1]
        #attn4 = self.attn4(d4_ecg , e2)
        cat_ecg = layers.concatenate([d4_ecg , e2])
        d5_ecg = self.de5_ecg(cat_ecg)
        d5_ecg = d5_ecg[:,:,:-1]
        #attn5 = self.attn5(d5_ecg , e1)
        cat_ecg = layers.concatenate([d5_ecg , e1])
        d6_ecg = self.de6_ecg(cat_ecg)[:,:,:-1]
        d7_ecg = self.de7_ecg(d6_ecg)
        d8_ecg = self.de8_ecg(d7_ecg)
        d9_ecg = self.de9_ecg(d8_ecg)
        #d10_ecg = self.ev1(d9_ecg)

        return  d9_ecg  #,attn1,attn2,attn3,attn4,attn5


class BRUnet_Multi_resp(tf.keras.Model):
    def __init__(self, in_channels):
        super(BRUnet_Multi_resp,self).__init__()

        self.en1 = keras.Sequential([layers.Conv1D(32,kernel_size = 3, padding = 'same',input_shape = in_channels),
                                    layers.BatchNormalization(axis = -1),
                                    layers.LeakyReLU(alpha = 0.2),
                                    layers.Conv1D(32, kernel_size = 5 , strides = 2 , padding = 'same'),
                                    IncBlock(32,32)])

        self.en2 = keras.Sequential([layers.Conv1D(64,kernel_size = 3,padding = 'same'),
                                     layers.BatchNormalization(axis = -1),
                                     layers.LeakyReLU(alpha = 0.2),
                                     layers.Conv1D(64, kernel_size=5, strides=2, padding='same'),
                                     IncBlock(64,64)])

        self.en3 = keras.Sequential([layers.Conv1D(128, kernel_size=3, padding='same'),
                                      layers.BatchNormalization(axis = -1),
                                      layers.LeakyReLU(alpha = 0.2),
                                      layers.Conv1D(128, kernel_size= 3, strides=2, padding = 'same'),
                                      IncBlock(128,128)])

        self.en4 = keras.Sequential([layers.Conv1D(256, kernel_size= 3, padding = 'same'),
                                     layers.BatchNormalization(axis = -1),
                                     layers.LeakyReLU(alpha = 0.2),
                                     layers.Conv1D(256 , kernel_size= 4, strides = 2 , padding = 'same'),
                                     IncBlock(256,256)])

        self.en5 = keras.Sequential([layers.Conv1D(512, kernel_size = 3, padding = 'same'),
                                     layers.BatchNormalization(axis = -1),
                                     layers.LeakyReLU(alpha = 0.2),
                                     layers.Conv1D(512 , kernel_size = 3 , padding = 'same'),
                                     IncBlock(512,512)])
        
        self.en6 = keras.Sequential([layers.Conv1D(1024, kernel_size=3 , padding = 'same'),
                                      layers.BatchNormalization(axis = -1),
                                      layers.LeakyReLU(alpha = 0.2),
                                      IncBlock(1024,1024)])

        #self.attn1 = AttentionBlock(1024)

        self.en7_p = keras.Sequential([layers.Conv1D(128 , kernel_size = 4 , strides = 2 , padding = 'same'),
                                       layers.BatchNormalization(axis = -1),
                                       layers.LeakyReLU(alpha = 0.2),
                                       IncBlock(128,128)])
        
        #self.attn2 = AttentionBlock(128)
        
        self.en8_p = keras.Sequential([layers.Conv1D(64 , kernel_size = 4 , strides = 2, padding = 'same'),
                                       layers.BatchNormalization(axis = -1),
                                       layers.LeakyReLU(alpha = 0.2),
                                       IncBlock(64,64)])
        
        #self.attn3 = AttentionBlock(64)

        self.en9_p = keras.Sequential([layers.Conv1D(4 , kernel_size = 4 , strides = 2, padding = 'same'),
                                      layers.BatchNormalization(axis = -1),
                                      layers.LeakyReLU(alpha = 0.2),
                                      IncBlock(4,4)])
        
        #self.attn4 = AttentionBlock(4)

        self.fc = layers.Dense(1)

        #self.ev1 = edl.layers.DenseNormalGamma(1)

        self.de1_ecg = keras.Sequential([layers.Conv1DTranspose(512, kernel_size = 1,strides = 1,padding = 'same'),
                                         layers.BatchNormalization(axis = -1),
                                         layers.LeakyReLU(alpha = 0.2),
                                         IncBlock(512,512)])

        #self.attn5 = AttentionBlock(512,512)
        
        
        self.de2_ecg = keras.Sequential([layers.Conv1DTranspose(512 , kernel_size= 3 , strides = 1,padding = 'same'),
                                        layers.BatchNormalization(axis = -1),
                                        layers.LeakyReLU(alpha=0.2),
                                        layers.Conv1DTranspose(256 , kernel_size= 4, strides = 1,padding = 'same'),
                                        IncBlock(256,256)])

        #self.attn6 = AttentionBlock(256,256)
        
        self.de3_ecg = keras.Sequential([layers.Conv1D(256, kernel_size=3,padding='same'),
                                         layers.BatchNormalization(axis = -1),
                                         layers.LeakyReLU(alpha = 0.2),
                                         layers.Conv1DTranspose(128 , kernel_size= 4 , strides=2,padding = 'same'),
                                         IncBlock(128,128)])
        
        #self.attn7 = AttentionBlock(128,128)

        self.de4_ecg = keras.Sequential([layers.Conv1D(128 , kernel_size=1 , strides=1 , padding='same'),
                                         layers.BatchNormalization(axis = -1),
                                         layers.LeakyReLU(alpha = 0.2),
                                         layers.Conv1DTranspose(64 , kernel_size= 1 , strides = 2,padding= 'same'),
                                         IncBlock(64,64)])
        
        #self.attn8 = AttentionBlock(64,64)

        self.de5_ecg = keras.Sequential([layers.Conv1D(64 , kernel_size= 3 , strides=1, padding='same'),
                                         layers.BatchNormalization(axis = -1),
                                         layers.LeakyReLU(alpha = 0.2),
                                         layers.Conv1DTranspose(32 , kernel_size=3, strides=2,padding = 'same'),
                                         IncBlock(32,32)])
        
        #self.attn9 = AttentionBlock(32,32)

        self.de6_ecg = keras.Sequential([layers.Conv1D(32 , kernel_size = 3 , strides = 1 , padding = 'same'),
                                        layers.BatchNormalization(axis = -1),
                                        layers.LeakyReLU(alpha = 0.2),
                                        layers.Conv1DTranspose(16 , kernel_size=3 , strides = 2, padding='same'),
                                        IncBlock(16,16)])

        self.de7_ecg = keras.Sequential([layers.Conv1DTranspose(1, kernel_size = 1,strides = 1,padding = 'same'),
                                         layers.LeakyReLU(alpha = 0.2)])

        self.de8_ecg = keras.Sequential([layers.Conv1DTranspose(1, kernel_size = 1,strides = 1,padding = 'same'),
                                         layers.LeakyReLU(alpha = 0.2)])
        
        self.de9_ecg = keras.Sequential([layers.Conv1DTranspose(1, kernel_size = 1,strides = 1,padding = 'same'),
                                         layers.LeakyReLU(alpha = 0.2)])

        #self.ev2 = edl.layers.DenseNormalGamma(1)
    
    def call(self,x, training = False):
        
        e1 = self.en1(x)
        e2 = self.en2(e1)
        e3 = self.en3(e2)
        e4 = self.en4(e3)
        e5 = self.en5(e4)
        e6 = self.en6(e5)
        #attn1 = self.attn1(e6)
        out_1 = self.en7_p(e6)
        #attn2 = self.attn2(out_1)
        out_2 = self.en8_p(out_1)
        #attn3 = self.attn3(out_2)
        out_3 = self.en9_p(out_2)
        #attn4 = self.attn4(out_3)
        out_4 = self.fc(out_3)

        d1_ecg = self.de1_ecg(e6)
        #attn5 = self.attn5(d1_ecg , e5)
        cat_ecg = layers.concatenate([d1_ecg,e5])
        d2_ecg = self.de2_ecg(cat_ecg)
        #attn6 = self.attn6(d2_ecg , e4)
        cat_ecg = layers.concatenate([d2_ecg,e4])
        d3_ecg = self.de3_ecg(cat_ecg)
        #attn7 = self.attn7(d3_ecg , e3)
        cat_ecg = layers.concatenate([d3_ecg,e3])
        d4_ecg = self.de4_ecg(cat_ecg)
        d4_ecg = d4_ecg[:,:,:-1]
        #attn8 = self.attn8(d4_ecg , e2)
        cat_ecg = layers.concatenate([d4_ecg,e2])
        d5_ecg = self.de5_ecg(cat_ecg)
        d5_ecg = d5_ecg[:,:,:-1]
        #attn9 = self.attn9(d5_ecg , e1)
        cat_ecg = layers.concatenate([d5_ecg,e1])
        d6_ecg = self.de6_ecg(cat_ecg)[:,:,:-1]
        d7_ecg = self.de7_ecg(d6_ecg)
        d8_ecg = self.de8_ecg(d7_ecg)
        d9_ecg = self.de9_ecg(d8_ecg)
        #d10_ecg = self.ev2(d9_ecg)

        return d9_ecg,out_4   #,attn1,attn2,attn3,attn4,attn5,attn6,attn7,attn8,attn9

class BRUnet_Encoder(tf.keras.Model):
    def __init__(self,in_channels):
        super(BRUnet_Encoder, self).__init__()

        self.en1 = keras.Sequential([layers.Conv1D(32,kernel_size = 3, padding = 'same', input_shape = in_channels),
                                    layers.BatchNormalization(axis = -1),
                                    layers.LeakyReLU(alpha = 0.2),
                                    layers.Conv1D(32, kernel_size = 5, strides = 2,padding = 'same'),
                                    IncBlock(32,32)])
        
        self.en2 = keras.Sequential([layers.Conv1D(64 , kernel_size = 3 , padding = 'same'),
                                      layers.BatchNormalization(axis = -1),
                                      layers.LeakyReLU(alpha = 0.2),
                                      layers.Conv1D(64 , kernel_size = 5 , strides = 2, padding = 'same'),
                                      IncBlock(64,64)])
        
        self.en3 = keras.Sequential([layers.Conv1D(128, kernel_size = 3 , padding = 'same'),
                                     layers.BatchNormalization(axis = -1),
                                     layers.LeakyReLU(alpha = 0.2),
                                     layers.Conv1D(128,kernel_size = 3 , strides = 2, padding = 'same'),
                                     IncBlock(128,128)])
        
        self.en4 = keras.Sequential([layers.Conv1D(256 , kernel_size = 3,padding = 'same'),
                                      layers.BatchNormalization(axis = -1),
                                      layers.LeakyReLU(alpha = 0.2),
                                      layers.Conv1D(256 , kernel_size = 4 , strides = 2 , padding = 'same'),
                                      IncBlock(256,256)])
        
        self.en5 = keras.Sequential([layers.Conv1D(512 , kernel_size = 3 , padding = 'same'),
                                     layers.BatchNormalization(axis = -1),
                                     layers.LeakyReLU(alpha = 0.2),
                                     layers.Conv1D(512 , kernel_size = 3 , padding = 'same'),
                                     IncBlock(512,512)])
        
        self.en6 = keras.Sequential([layers.Conv1D(1024, kernel_size = 3 , padding = 'same'),
                                      layers.BatchNormalization(axis = -1),
                                      layers.LeakyReLU(alpha = 0.2),
                                      IncBlock(1024,1024)])
        
        #self.attn1 = AttentionBlock(1024)

        self.en7_p = keras.Sequential([layers.Conv1D(128 , kernel_size = 4, strides = 2 , padding = 'same'),
                                       layers.BatchNormalization(axis = -1),
                                       layers.LeakyReLU(alpha = 0.2),
                                       IncBlock(128,128)])
        
        #self.attn2 = AttentionBlock(128)

        self.en8_p = keras.Sequential([layers.Conv1D(64 , kernel_size = 4 , strides = 2, padding = 'same'),
                                       layers.BatchNormalization(axis = -1),
                                       layers.LeakyReLU(alpha = 0.2),
                                       IncBlock(64,64)])
        
        #self.attn3 = AttentionBlock(64)

        self.en9_p = keras.Sequential([layers.Conv1D(4 , kernel_size = 4, strides = 2 , padding = 'same'),
                                        layers.BatchNormalization(axis = -1),
                                        layers.LeakyReLU(alpha = 0.2),
                                        IncBlock(4,4)])
        #self.attn4 = AttentionBlock(4)
        self.fc = layers.Dense(1)

        #self.ev1 = edl.layers.DenseNormalGamma(1)

    def call(self,x, training = False):
        #import pdb;pdb.set_trace()
        e1 = self.en1(x)
        e2 = self.en2(e1)
        e3 = self.en3(e2)
        e4 = self.en4(e3)
        e5 = self.en5(e4)
        e6 = self.en6(e5)
        #attn1 = self.attn1(e6)
        out_1 = self.en7_p(e6)
        #attn2 = self.attn2(out_1)
        out_2 = self.en8_p(out_1)
        #attn3 = self.attn3(out_2)
        out_3 = self.en9_p(out_2)
        #attn4 = self.attn4(out_3)
        out_4 = self.fc(out_3)
        #out_5 = self.ev1(out_4)
        return out_4  #,attn1,attn2,attn3,attn4

class BRUnet_raw(tf.keras.Model):
    def __init__(self, in_channels):
        super(BRUnet_raw,self).__init__()
        #in_channels = shape[1]

        self.layer1 = Subsampling(in_channels)
        
        self.en1 = keras.Sequential([layers.Conv1D(32,kernel_size = 3, padding = 'same',input_shape = in_channels),
                                    layers.BatchNormalization(axis = -1),
                                    layers.LeakyReLU(alpha = 0.2),
                                    layers.Conv1D(32, kernel_size = 5 , strides = 2 , padding = 'same'),
                                    IncBlock(32,32)])

        self.en2 = keras.Sequential([layers.Conv1D(64, kernel_size = 3 , padding = 'same'),
                                     layers.BatchNormalization(axis = -1),
                                     layers.LeakyReLU(alpha = 0.2),
                                     layers.Conv1D(64 , kernel_size = 5,strides = 2, padding = 'same'),
                                     IncBlock(64,64)])

        self.en3 = keras.Sequential([layers.Conv1D(128, kernel_size = 3 , padding = 'same'),
                                     layers.BatchNormalization(axis = -1),
                                     layers.LeakyReLU(alpha = 0.2),
                                     layers.Conv1D(128 , kernel_size = 3,strides = 2, padding = 'same'),
                                     IncBlock(128,128)])

        self.en4 = keras.Sequential([layers.Conv1D(256, kernel_size = 3 , padding = 'same'),
                                     layers.BatchNormalization(axis = -1),
                                     layers.LeakyReLU(alpha = 0.2),
                                     layers.Conv1D(256 , kernel_size = 4,strides = 2,padding = 'same'),
                                     IncBlock(256,256)])

        self.en5 = keras.Sequential([layers.Conv1D(512, kernel_size = 3 , padding = 'same'),
                                     layers.BatchNormalization(axis = -1),
                                     layers.LeakyReLU(alpha = 0.2),
                                     layers.Conv1D(512 , kernel_size = 4,strides = 2,padding = 'same'),
                                     IncBlock(512,512)])

        self.en6 = keras.Sequential([layers.Conv1D(1024, kernel_size = 3 , padding = 'same'),
                                     layers.BatchNormalization(axis = -1),
                                     layers.LeakyReLU(alpha = 0.2),
                                     IncBlock(1024,1024)])
        
        self.de1_ecg = keras.Sequential([layers.Conv1DTranspose(512, kernel_size = 1,strides = 1 , padding = 'same'),
                                         layers.BatchNormalization(axis = -1),
                                         layers.LeakyReLU(alpha = 0.2),
                                         IncBlock(512,512)])
        #self.attn1 = AttentionBlock(512,512)

        self.de2_ecg = keras.Sequential([layers.Conv1DTranspose(512, kernel_size = 1,strides = 2,padding = 'same'),
                                         layers.BatchNormalization(axis = -1),
                                         layers.LeakyReLU(alpha = 0.2),
                                         layers.Conv1DTranspose(256 , kernel_size = 1,strides = 1,padding = 'same'),
                                         IncBlock(256,256)])
        
        #self.attn2 = AttentionBlock(256, 256)

        self.de3_ecg = keras.Sequential([layers.Conv1D(256, kernel_size = 1,strides = 1),  #kernel_size = 3
                                         layers.BatchNormalization(axis = -1),
                                         layers.LeakyReLU(alpha = 0.2),
                                         layers.Conv1DTranspose(128 , kernel_size = 4, strides = 2, padding = 'same'), #kernel_size = 4 strides = 2
                                         IncBlock(128,128)])
        
        #self.attn3 = AttentionBlock(128, 128)

        self.de4_ecg = keras.Sequential([layers.Conv1D(128, kernel_size = 1,strides = 1), #kernel_size = 3
                                         layers.BatchNormalization(axis = -1),
                                         layers.LeakyReLU(alpha = 0.2),
                                         layers.Conv1DTranspose(64 , kernel_size = 3,strides = 2,padding = 'same'),  #kernel_size = 3 strides = 2
                                         IncBlock(64,64)])
        
        #self.attn4 = AttentionBlock(64, 64)

        self.de5_ecg = keras.Sequential([layers.Conv1D(64, kernel_size = 1,strides = 1,padding = 'same'), #kernel_size = 3
                                         layers.BatchNormalization(axis = -1),   
                                         layers.LeakyReLU(alpha = 0.2),
                                         layers.Conv1DTranspose(32 , kernel_size = 3, strides = 2,padding = 'same'), #kernel_size = 3,strides = 2
                                         IncBlock(32,32)])
        
        #self.attn5 = AttentionBlock(32, 32)

        self.de6_ecg = keras.Sequential([layers.Conv1D(32, kernel_size = 1, strides = 1), #kernel_size = 3
                                         layers.BatchNormalization(axis = -1),
                                         layers.LeakyReLU(alpha = 0.2),
                                         layers.Conv1DTranspose(16 , kernel_size = 3,strides = 2,padding = 'same'), #kernel_size = 3,strides = 2
                                         IncBlock(16,16)])
        
        self.de7_ecg = keras.Sequential([layers.Conv1DTranspose(1, kernel_size = 1,strides = 1,padding = 'same'),
                                         layers.LeakyReLU(alpha = 0.2)])

        self.de8_ecg = keras.Sequential([layers.Conv1DTranspose(1, kernel_size = 1,strides = 1,padding = 'same'),
                                         layers.LeakyReLU(alpha = 0.2)])
        
        self.de9_ecg = keras.Sequential([layers.Conv1DTranspose(1, kernel_size = 1,strides = 1,padding = 'same'),
                                         layers.LeakyReLU(alpha = 0.2)])
        
    def call (self,x,training = False):
        #import pdb;pdb.set_trace()
        l1 = self.layer1(x)
        e1 = self.en1(l1)
        e2 = self.en2(e1)
        e3 = self.en3(e2)
        e4 = self.en4(e3)
        e5 = self.en5(e4)
        e6 = self.en6(e5)
        d1_ecg = self.de1_ecg(e6)
        #attn1 = self.attn1(d1_ecg , e5)
        cat_ecg = layers.concatenate([d1_ecg , e5])
        d2_ecg = self.de2_ecg(cat_ecg)
        #attn2 = self.attn2(d2_ecg , e4)
        cat_ecg = layers.concatenate([d2_ecg , e4])
        d3_ecg = self.de3_ecg(cat_ecg)
        #attn3 = self.attn3(d3_ecg , e3)
        cat_ecg = layers.concatenate([d3_ecg , e3])
        #import pdb;pdb.set_trace()
        d4_ecg = self.de4_ecg(cat_ecg)
        d4_ecg = d4_ecg[:,:,:-1]
        #attn4 = self.attn4(d4_ecg , e2)
        cat_ecg = layers.concatenate([d4_ecg , e2])
        d5_ecg = self.de5_ecg(cat_ecg)
        d5_ecg = d5_ecg[:,:,:-1]
        #attn5 = self.attn5(d5_ecg , e1)
        cat_ecg = layers.concatenate([d5_ecg , e1])
        d6_ecg = self.de6_ecg(cat_ecg)[:,:,:-1]
        d7_ecg = self.de7_ecg(d6_ecg)
        d8_ecg = self.de8_ecg(d7_ecg)
        d9_ecg = self.de9_ecg(d8_ecg)
        #d10_ecg = self.ev1(d9_ecg)

        return  d9_ecg


class BRUnet_raw_encoder(tf.keras.Model):
    def __init__(self,in_channels):
        super(BRUnet_raw_encoder, self).__init__()

        self.layer1 = Subsampling(in_channels)

        self.en1 = keras.Sequential([layers.Conv1D(32,kernel_size = 3, padding = 'same', input_shape = in_channels),
                                    layers.BatchNormalization(axis = -1),
                                    layers.LeakyReLU(alpha = 0.2),
                                    layers.Conv1D(32, kernel_size = 5, strides = 2,padding = 'same'),
                                    IncBlock(32,32)])
        
        self.en2 = keras.Sequential([layers.Conv1D(64 , kernel_size = 3 , padding = 'same'),
                                      layers.BatchNormalization(axis = -1),
                                      layers.LeakyReLU(alpha = 0.2),
                                      layers.Conv1D(64 , kernel_size = 5 , strides = 2, padding = 'same'),
                                      IncBlock(64,64)])
        
        self.en3 = keras.Sequential([layers.Conv1D(128, kernel_size = 3 , padding = 'same'),
                                     layers.BatchNormalization(axis = -1),
                                     layers.LeakyReLU(alpha = 0.2),
                                     layers.Conv1D(128,kernel_size = 3 , strides = 2, padding = 'same'),
                                     IncBlock(128,128)])
        
        self.en4 = keras.Sequential([layers.Conv1D(256 , kernel_size = 3,padding = 'same'),
                                      layers.BatchNormalization(axis = -1),
                                      layers.LeakyReLU(alpha = 0.2),
                                      layers.Conv1D(256 , kernel_size = 4 , strides = 2 , padding = 'same'),
                                      IncBlock(256,256)])
        
        self.en5 = keras.Sequential([layers.Conv1D(512 , kernel_size = 3 , padding = 'same'),
                                     layers.BatchNormalization(axis = -1),
                                     layers.LeakyReLU(alpha = 0.2),
                                     layers.Conv1D(512 , kernel_size = 3 , padding = 'same'),
                                     IncBlock(512,512)])
        
        self.en6 = keras.Sequential([layers.Conv1D(1024, kernel_size = 3 , padding = 'same'),
                                      layers.BatchNormalization(axis = -1),
                                      layers.LeakyReLU(alpha = 0.2),
                                      IncBlock(1024,1024)])
        
        #self.attn1 = AttentionBlock(1024)

        self.en7_p = keras.Sequential([layers.Conv1D(128 , kernel_size = 4, strides = 2 , padding = 'same'),
                                       layers.BatchNormalization(axis = -1),
                                       layers.LeakyReLU(alpha = 0.2),
                                       IncBlock(128,128)])
        
        #self.attn2 = AttentionBlock(128)

        self.en8_p = keras.Sequential([layers.Conv1D(64 , kernel_size = 4 , strides = 2, padding = 'same'),
                                       layers.BatchNormalization(axis = -1),
                                       layers.LeakyReLU(alpha = 0.2),
                                       IncBlock(64,64)])
        
        #self.attn3 = AttentionBlock(64)

        self.en9_p = keras.Sequential([layers.Conv1D(4 , kernel_size = 4, strides = 2 , padding = 'same'),
                                        layers.BatchNormalization(axis = -1),
                                        layers.LeakyReLU(alpha = 0.2),
                                        IncBlock(4,4)])
        #self.attn4 = AttentionBlock(4)
        self.fc = layers.Dense(1)

        #self.ev1 = edl.layers.DenseNormalGamma(1)

    def call(self,x, training = False):
        #import pdb;pdb.set_trace()
        l1 = self.layer1(x)
        e1 = self.en1(l1)
        e2 = self.en2(e1)
        e3 = self.en3(e2)
        e4 = self.en4(e3)
        e5 = self.en5(e4)
        e6 = self.en6(e5)
        #attn1 = self.attn1(e6)
        out_1 = self.en7_p(e6)
        #attn2 = self.attn2(out_1)
        out_2 = self.en8_p(out_1)
        #attn3 = self.attn3(out_2)
        out_3 = self.en9_p(out_2)
        #attn4 = self.attn4(out_3)
        out_4 = self.fc(out_3)
        #out_5 = self.ev1(out_4)
        return out_4   #,attn1,attn2,attn3,attn4

class BRUnet_raw_multi(tf.keras.Model):
    def __init__(self, in_channels):
        super(BRUnet_raw_multi,self).__init__()

        self.layer1 = Subsampling(in_channels)

        self.en1 = keras.Sequential([layers.Conv1D(32,kernel_size = 3, padding = 'same',input_shape = in_channels),
                                    layers.BatchNormalization(axis = -1),
                                    layers.LeakyReLU(alpha = 0.2),
                                    layers.Conv1D(32, kernel_size = 5 , strides = 2 , padding = 'same'),
                                    IncBlock(32,32)])

        self.en2 = keras.Sequential([layers.Conv1D(64,kernel_size = 3,padding = 'same'),
                                     layers.BatchNormalization(axis = -1),
                                     layers.LeakyReLU(alpha = 0.2),
                                     layers.Conv1D(64, kernel_size=5, strides=2, padding='same'),
                                     IncBlock(64,64)])

        self.en3 = keras.Sequential([layers.Conv1D(128, kernel_size=3, padding='same'),
                                      layers.BatchNormalization(axis = -1),
                                      layers.LeakyReLU(alpha = 0.2),
                                      layers.Conv1D(128, kernel_size= 3, strides=2, padding = 'same'),
                                      IncBlock(128,128)])

        self.en4 = keras.Sequential([layers.Conv1D(256, kernel_size= 3, padding = 'same'),
                                     layers.BatchNormalization(axis = -1),
                                     layers.LeakyReLU(alpha = 0.2),
                                     layers.Conv1D(256 , kernel_size= 4, strides = 2 , padding = 'same'),
                                     IncBlock(256,256)])

        self.en5 = keras.Sequential([layers.Conv1D(512, kernel_size = 3, padding = 'same'),
                                     layers.BatchNormalization(axis = -1),
                                     layers.LeakyReLU(alpha = 0.2),
                                     layers.Conv1D(512 , kernel_size = 3 , padding = 'same'),
                                     IncBlock(512,512)])
        
        self.en6 = keras.Sequential([layers.Conv1D(1024, kernel_size=3 , padding = 'same'),
                                      layers.BatchNormalization(axis = -1),
                                      layers.LeakyReLU(alpha = 0.2),
                                      IncBlock(1024,1024)])

        #self.attn1 = AttentionBlock(1024)

        self.en7_p = keras.Sequential([layers.Conv1D(128 , kernel_size = 4 , strides = 2 , padding = 'same'),
                                       layers.BatchNormalization(axis = -1),
                                       layers.LeakyReLU(alpha = 0.2),
                                       IncBlock(128,128)])
        
        #self.attn2 = AttentionBlock(128)
        
        self.en8_p = keras.Sequential([layers.Conv1D(64 , kernel_size = 4 , strides = 2, padding = 'same'),
                                       layers.BatchNormalization(axis = -1),
                                       layers.LeakyReLU(alpha = 0.2),
                                       IncBlock(64,64)])
        
        #self.attn3 = AttentionBlock(64)

        self.en9_p = keras.Sequential([layers.Conv1D(4 , kernel_size = 4 , strides = 2, padding = 'same'),
                                      layers.BatchNormalization(axis = -1),
                                      layers.LeakyReLU(alpha = 0.2),
                                      IncBlock(4,4)])
        
        #self.attn4 = AttentionBlock(4)

        self.fc = layers.Dense(1)

        #self.ev1 = edl.layers.DenseNormalGamma(1)

        self.de1_ecg = keras.Sequential([layers.Conv1DTranspose(512, kernel_size = 1,strides = 1,padding = 'same'),
                                         layers.BatchNormalization(axis = -1),
                                         layers.LeakyReLU(alpha = 0.2),
                                         IncBlock(512,512)])

        #self.attn5 = AttentionBlock(512,512)
        
        
        self.de2_ecg = keras.Sequential([layers.Conv1DTranspose(512 , kernel_size= 3 , strides = 1,padding = 'same'),
                                        layers.BatchNormalization(axis = -1),
                                        layers.LeakyReLU(alpha=0.2),
                                        layers.Conv1DTranspose(256 , kernel_size= 4, strides = 1,padding = 'same'),
                                        IncBlock(256,256)])

        #self.attn6 = AttentionBlock(256,256)
        
        self.de3_ecg = keras.Sequential([layers.Conv1D(256, kernel_size=3,padding='same'),
                                         layers.BatchNormalization(axis = -1),
                                         layers.LeakyReLU(alpha = 0.2),
                                         layers.Conv1DTranspose(128 , kernel_size= 4 , strides=2,padding = 'same'),
                                         IncBlock(128,128)])
        
        #self.attn7 = AttentionBlock(128,128)

        self.de4_ecg = keras.Sequential([layers.Conv1D(128 , kernel_size=1 , strides=1 , padding='same'),
                                         layers.BatchNormalization(axis = -1),
                                         layers.LeakyReLU(alpha = 0.2),
                                         layers.Conv1DTranspose(64 , kernel_size= 1 , strides = 2,padding= 'same'),
                                         IncBlock(64,64)])
        
        #self.attn8 = AttentionBlock(64,64)

        self.de5_ecg = keras.Sequential([layers.Conv1D(64 , kernel_size= 3 , strides=1, padding='same'),
                                         layers.BatchNormalization(axis = -1),
                                         layers.LeakyReLU(alpha = 0.2),
                                         layers.Conv1DTranspose(32 , kernel_size=3, strides=2,padding = 'same'),
                                         IncBlock(32,32)])
        
        #self.attn9 = AttentionBlock(32,32)

        self.de6_ecg = keras.Sequential([layers.Conv1D(32 , kernel_size = 3 , strides = 1 , padding = 'same'),
                                        layers.BatchNormalization(axis = -1),
                                        layers.LeakyReLU(alpha = 0.2),
                                        layers.Conv1DTranspose(16 , kernel_size=3 , strides = 2, padding='same'),
                                        IncBlock(16,16)])

        self.de7_ecg = keras.Sequential([layers.Conv1DTranspose(1, kernel_size = 1,strides = 1,padding = 'same'),
                                         layers.LeakyReLU(alpha = 0.2)])

        self.de8_ecg = keras.Sequential([layers.Conv1DTranspose(1, kernel_size = 1,strides = 1,padding = 'same'),
                                         layers.LeakyReLU(alpha = 0.2)])
        
        self.de9_ecg = keras.Sequential([layers.Conv1DTranspose(1, kernel_size = 1,strides = 1,padding = 'same'),
                                         layers.LeakyReLU(alpha = 0.2)])
    
    def call(self,x, training = False):
        l1 = self.layer1(x)
        e1 = self.en1(l1)
        e2 = self.en2(e1)
        e3 = self.en3(e2)
        e4 = self.en4(e3)
        e5 = self.en5(e4)
        e6 = self.en6(e5)
        #attn1 = self.attn1(e6)
        out_1 = self.en7_p(e6)
        #attn2 = self.attn2(out_1)
        out_2 = self.en8_p(out_1)
        #attn3 = self.attn3(out_2)
        out_3 = self.en9_p(out_2)
        #attn4 = self.attn4(out_3)
        out_4 = self.fc(out_3)

        d1_ecg = self.de1_ecg(e6)
        #attn5 = self.attn5(d1_ecg , e5)
        cat_ecg = layers.concatenate([d1_ecg,e5])
        d2_ecg = self.de2_ecg(cat_ecg)
        #attn6 = self.attn6(d2_ecg , e4)
        cat_ecg = layers.concatenate([d2_ecg,e4])
        d3_ecg = self.de3_ecg(cat_ecg)
        #attn7 = self.attn7(d3_ecg , e3)
        cat_ecg = layers.concatenate([d3_ecg,e3])
        d4_ecg = self.de4_ecg(cat_ecg)
        d4_ecg = d4_ecg[:,:,:-1]
        #attn8 = self.attn8(d4_ecg , e2)
        cat_ecg = layers.concatenate([d4_ecg,e2])
        d5_ecg = self.de5_ecg(cat_ecg)
        d5_ecg = d5_ecg[:,:,:-1]
        #attn9 = self.attn9(d5_ecg , e1)
        cat_ecg = layers.concatenate([d5_ecg,e1])
        d6_ecg = self.de6_ecg(cat_ecg)[:,:,:-1]
        d7_ecg = self.de7_ecg(d6_ecg)
        d8_ecg = self.de8_ecg(d7_ecg)
        d9_ecg = self.de9_ecg(d8_ecg)
        #d10_ecg = self.ev2(d9_ecg)

        return d9_ecg,out_4   #,attn1,attn2,attn3,attn4,attn5,attn6,attn7,attn8,attn9

 




        
