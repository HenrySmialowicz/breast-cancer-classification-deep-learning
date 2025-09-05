#Imports
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import SeparableConv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class CancerNet:
  #building CNN
  @staticmethod
  def build(width,height,depth,classes):
    #Creates linear stack of layers and defines input dimensions of images
    model=Sequential()
    shape=(height,width,depth)
    channelDim=-1
                            
    if K.image_data_format()=="channels_first":
      shape=(depth,height,width)
      channelDim=1
                
    #Depthwise seperable convolution - lighter and faster
    model.add(SeparableConv2D(32, (3,3), padding="same",input_shape=shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channelDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    #Two layers with 64 filters each - edges, textures
    model.add(SeparableConv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channelDim))
    model.add(SeparableConv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channelDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
          
    #Three layers with 128 filters each - shapes, patterns
    model.add(SeparableConv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channelDim))
    model.add(SeparableConv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channelDim))
    model.add(SeparableConv2D(128, (3,3), padding="same"))              
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channelDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
      
    #Flattened into long vector and passed through dense 'decision making' layer
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
          
    model.add(Dense(classes))
    model.add(Activation("softmax"))
        
    return model
