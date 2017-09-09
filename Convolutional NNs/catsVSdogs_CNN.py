
"""
Image Classificatioon
"""

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

# Initialize the CNN - Sequential layers:
classifier = Sequential()


#*************************************
#        STEP 1: CONVOLUTION  
#*************************************
# Add Convolution layer to 'classifier' sequence:

'''
*'Convolution Kernel' ~ 'feature detector'
Convolution2D(32, 3, 3)   ----->  create 32 feature detectors of dimension 3x3.
border_mode='same'        ----->  how to handle borders
input_shape(64, 64, 3)    ----->  how our images will be converted (3D array for color, 2D for B&W)
                                  64, 64 for array size, 3 for color image. In essence, the kind of put we expect.
activation = 'relu'       ----->  activation function to be used during the convolution.
'''
classifier.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(64, 64, 3), activation = 'relu'))


#*************************************
#          STEP 2: POOLING  
#*************************************
# Create a feature map by pooling. Reduce the number of nodes for the next step (flattening).
# Pooling allows us to extract spatial structure and other important features. 

# Add MaxPooling to to 'classifier' sequence:
'''
pool_size=(2,2)           ------> size of feature map -- 2,2 = reduce by 2
'''
classifier.add(MaxPooling2D(pool_size=(2,2)))


#*************************************
#          STEP 3: FLATTEN  
#*************************************
# Get maps ready for input into 'fully connected' layer.
# Turns it into 'input layer' of fully connected layer.
classifier.add(Flatten())


#*************************************
#       STEP 4: FULL CONNECTION  
#*************************************
# Feed 'flattened' feature maps into fully connected layer.
'''
output_dim = 128       -----> number of nodes in fully connected hidden layer.
activation='relu'      -----> activation function applied in the 1st layer hidden nodes.

output_dim = 1         -----> only one output (this is the final output)
activation='sigmoid'   -----> activation function applied to output layer. Sgmoid because our 
                              outcome is binary (cats VS dogs). If more than two categories,
                              choose 'softmax'.
'''
# Two hidden layers: 
classifier.add(Dense(output_dim = 128, activation='relu'))
classifier.add(Dense(output_dim = 1, activation='sigmoid'))


#*************************************
#           COMPILE CNN  
#*************************************
'''
loss='binary_crossentropy'       -----> because we have a binary classification problem. Use
                                        'categorical' cross-entropy for mult. features.
'''
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


#=================================================================
#                     IMAGE AUGMENTATION:
#=================================================================
# Apply several performations to prevent overfitting.
from keras.preprocessing.image import ImageDataGenerator

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
'''
'dataset/training_set'       ------> where we are extracting training images from.
target_size = (64, 64)       ------> size of images expected by our CNN.
                                     Must match the same number from step 1 -- input_shape=(64, 64, 3)
batch_size = 32              ------> number of images to put through the CNN before updating weights.
class_mode = 'binary'        ------> because our training class is binary.
'''
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Set Test set:
'''
'dataset/test_set'          ------> where we are extracting test images from.
'''
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


#=================================================================
#                    FIT MODEL TO DATA:
#=================================================================
'''
steps_per_epoch = 8000      -------> number of images in our training set. All must pass through the CNN
                                     during each epoch.
validation_data = test_set  -------> which set to validate performance against.
validation_steps = 2000     -------> number of images in our test set.
'''
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)




# PRINT RUN-TIME:
print("--- %s seconds ---" % (time.time() - start_time))



#=================================================================
#                        SAVE MODEL:
#=================================================================
from keras.models import load_model

classifier.save('catsVSdogs_CNN.h5')

del classifier
my_model = load_model('catsVSdogs_CNN.h5')


