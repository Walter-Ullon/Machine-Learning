#===============================================================
#                       TIME CODE:
#===============================================================
import time
start_time = time.time()


#=================================================================
#              BUILD CONCOLUTIONAL NEURAL NETWORK:
#=================================================================
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


#*************************************
#        MODEL PARAMETERS:  
#*************************************
img_width, img_height = 150, 150 
input_shape=(img_width, img_height, 3)
dropout_rate = 0.6


# Define Model Building function: 
def create_model(p, input_shape):
    '''
    *'Convolution Kernel' ~ 'feature detector'
    Convolution2D(32, 3, 3)   ----->  create 32 feature detectors of dimension 3x3.
    border_mode='same'        ----->  how to handle borders
    input_shape(32, 32, 3)    ----->  how our images will be converted (3D array for color, 2D for B&W)
                                      32, 32 for array size, 3 for color image. In essence, the kind of put we expect.
    activation = 'relu'       ----->  activation function to be used during the convolution.
    pool_size=(2,2)           ------> size of feature map -- 2,2 = reduce by 2
    '''
    # Initialize the CNN - Sequential layers:
    model = Sequential()
    #*************************************
    #   CONVOLUTION + POOLING LAYER #1 
    #*************************************
    model.add(Convolution2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #*************************************
    #   CONVOLUTION + POOLING LAYER #2 
    #************************************* 
    model.add(Convolution2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #*************************************
    #   CONVOLUTION + POOLING LAYER #3 
    #*************************************
    model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #*************************************
    #   CONVOLUTION + POOLING LAYER #4 
    #*************************************
    model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #*************************************
    #              FLATTEN  
    #*************************************
    # Get maps ready for input into 'fully connected' layer.
    # Turns it into 'input layer' of fully connected layer.
    model.add(Flatten())
    
    #*************************************
    #           FULL CONNECTION  
    #*************************************
    # Feed 'flattened' feature maps into fully connected layer.
    
    # Three hidden layers
    model.add(Dense(output_dim=64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(output_dim=64, activation='relu'))
    model.add(Dense(output_dim=64, activation='relu'))
    model.add(Dropout(dropout_rate/2))
    # Output layer:
    model.add(Dense(1, activation='sigmoid'))
    
    #*************************************
    #           COMPILE CNN  
    #*************************************
    optimizer = Adam(lr=1e-3)
    metrics=['accuracy']
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
    return model



#=================================================================
#                     IMAGE AUGMENTATION:
#=================================================================
# Apply several performations to prevent overfitting.
def run_training(bs, epochs):
    # Applies geometrical transformations:
    '''
    rescale = 1./255             ------> rescale all pixel values between 0-1.
    shear_range = 0.2            ------> random transvexions.
    zoom_range = 0.2             ------> random zoom.
    horizontal_flip = True       ------> flip images so that we dont find the same images in the batches.
    '''
    train_datagen = ImageDataGenerator(rescale = 1./255, 
                                       shear_range = 0.2, 
                                       zoom_range = 0.2, 
                                       horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    # Set Training set:
    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (img_width, img_height),
                                                 batch_size = bs,
                                                 class_mode = 'binary')
     
    # Set Test set:
    '''
    'dataset/test_set'          ------> where we are extracting test images from.
    '''                                        
    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (img_width, img_height),
                                            batch_size = bs,
                                            class_mode = 'binary')
    
    # Call model building function:                                        
    model = create_model(dropout_rate, input_shape=(img_width, img_height, 3))                                  
    # Fit model to data:
    model.fit_generator(training_set,
                         steps_per_epoch=8000/bs,
                         epochs = epochs,
                         validation_data = test_set,
                         validation_steps = 2000/bs)




#=================================================================
#                     RUN TRAINING SEQUENCE:
#=================================================================
# run_training(bs, epochs)
run_training(32, 100)



# PRINT RUN-TIME:
print("--- %s seconds ---" % (time.time() - start_time))





#=================================================================
#                        SAVE MODEL:
#=================================================================
from keras.models import load_model
model.save('catsVSdogs_CNN.h5')



#=================================================================
#                        LOAD MODEL:
#=================================================================
#my_model = load_model('catsVSdogs_CNN.h5')


'''
#=================================================================
#                  MAKE NEW PREDICTIONS:
#=================================================================
import numpy as np
from keras.preprocessing import image

# Load image:
test_image = image.load_img('dataset/single_prediction/cat_or_dog_10.jpeg', target_size = (64, 64))

# We make the image the same size and dimension as the model input:
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)


# Predict:
result = my_model.predict(test_image)

# Get class names corresponding to prediction output:
training_set.class_indices

# Define prediction result function: 
def get_prediction(pred):
    if result[0][0] == 1:
        pred = 'dog!'
    else:
        pred = 'cat!'    
    print('The animal is a ' + pred)

get_prediction(result)

'''















