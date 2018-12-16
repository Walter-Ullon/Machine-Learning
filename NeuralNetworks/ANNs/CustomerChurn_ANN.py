"""
Created on Mon Sep  4 08:08:07 2017
@author: walterullon
"""

'''
Geographical segmentation ANN model applied to customer churn problem. 
'''

#===============================================================
#                       TIME CODE:
#===============================================================
import time
start_time = time.time()


#=================================================================
#                      IMPORT PACKAGES:
#=================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#=================================================================
#                       LOAD DATA:
#=================================================================
churn = pd.read_csv('Churn_Modelling.csv')


#=================================================================
#                       CLEAN DATA:
#=================================================================
# Drop unnecesary features:
churn = churn.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

#=================================================================
#                HANDLE CATEGORICAL DATA:
#=================================================================
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

# Create encoder instances:
encoder = LabelBinarizer()
L_encoder = LabelEncoder()

# Handle Gender. Encode by label and create new dataframe:
gender = L_encoder.fit_transform(churn['Gender'])
gender = pd.DataFrame(gender, columns=['Male'])

# Handle Geography. Use LabelBinarizer to return 'one-hot' vectors:
geography = encoder.fit_transform(churn['Geography'])
geography = pd.DataFrame(geography, columns=encoder.classes_)
# Drop one geo category to avoid the 'dummy variable trap':
geography = geography.drop('Germany', axis=1)

# Concatenate binarized categorical data with original dataframe and drop old categories:
churn['Male'] = gender
churn[['France', 'Spain']] = geography
churn = churn.drop(['Gender','Geography'], axis=1)


#=================================================================
#                          SCALE DATA:
#=================================================================
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Create new array of scaled features. Save them to a new df for easier processing:
churn_scaled = scaler.fit_transform(churn)
churn_scaled = pd.DataFrame(churn_scaled, columns=[churn.columns])

#=================================================================
#                SET PREDICTOR AND TARGET CLASSES:
#=================================================================
# Everything except for 'Exited'...
X = churn_scaled.iloc[:, churn_scaled.columns != 'Exited' ]

# WARNING: make sure NOT to scale the target class. Take 'y' from original df instead..
y = churn['Exited']


#=================================================================
#                          SPLIT DATA:
#=================================================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 8)

# Change training and test set data type from dataframe to numpy array, else keras yells at you:
X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
# No need to convert y_test...

#=================================================================
#                BUILD ARTIFICIAL NEURAL NETWORK:
#=================================================================
# Import Keras and modules:
from keras.models import Sequential
from keras.layers import Dense, Dropout

#****************************
#      Set classifier:  
#****************************
classifier = Sequential()

#*************************************
# Add input layer & 1st hidden layer:  
#*************************************
'''
TIP: use the average of the # of ind. variables and the # of dependent variables to choose
     the number of nodes in hiddden layers.  
     
     units = 6                          ---> # of nodes in hidden layer.
     kernel_initializer = 'uniform'     ---> initial weights from an uniform dist.
     activation = 'relu'                ---> 'relu' activation function.
     input_dim = 11                     ---> # of nodes in unput layer, i.e. 11 ind. variables.
                                             We need to tell the network what inputs to expect.
'''
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
#classifier.add(Dropout(rate = 0.1))

''' 
We will apply 'dropout' to this layer (this helps to avoiding overfitting). By setting rate = 0.1, 
we are indicating that we want ten percent of the neurons disabled. We are in essence making
neurons less dependent on each other.
'''

#*************************************
#     Add 2nd hidden layer:  
#*************************************
'''
input_dim = 11           ---> no need for this on 2nd hidden layer as it will take 
                              as input the output of the 1st hidden layer. 
'''
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
#classifier.add(Dropout(rate = 0.1))

#*************************************
#        Add output layer:  
#*************************************
'''
output_dim=1             ---> we only want one node in the output layer.
activation = 'sigmoid'   ---> we want a sigmoid activation function since 
                              we'd like to return probabilities.
'''
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))


#*************************************
#          Compile ANN:  
#*************************************
'''
optimizer = 'adam'                 ---> optimizer to use to find the ideal weights.
loss = 'binary_crossentropy'       ---> loss function to use to calculate costs.
                                        'Logarithmic loss' is used for sigmoids.
                                        For binary outcomes use: 'binary_crossentropy'.
                                        For multiple categories use: 'categorical_crossentropy'.
metrics=['accuracy']               ---> criterion used to evaluate model, in this case 'accuracy'.                                                                     
'''
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


#=================================================================
#                TRAIN ARTIFICIAL NEURAL NETWORK:
#=================================================================
'''
batch_size = 10                ---> how many observations to go through before
                                    updating the weights.
nb_epoch=100                   ---> how many rounds we go through the steps of 
                                    updating weights, calculating inouts/outputs, etc..
'''
classifier.fit(X_train, y_train, batch_size=10, epochs=20)


#=================================================================
#                            PREDICT:
#=================================================================
# Returns probabilities:
predictions_raw = classifier.predict(X_test)

# Turn probabilities into binary results:
predictions = (predictions_raw > 0.5)


#=================================================================
#                 PREDICT A SINGLE OBSERVATION:
#=================================================================
"""
Predict if the customer with the following information will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000
"""
new_prediction = classifier.predict(scaler.fit_transform(np.array([[600, 40, 3, 60000, 2, 1, 1, 50000, 1, 1, 0]])))
new_prediction = (new_prediction > 0.5)


#=================================================================
#                          EVALUATE:
#=================================================================
from sklearn.metrics import confusion_matrix,classification_report
import itertools

print(classification_report(y_test, predictions))



CM = confusion_matrix(y_test, predictions)


# Get class names by calling unique values in 'priorityLevel' column.
class_names = list(churn['Exited'].unique())
class_names.sort()

# Confusion matrix plot function:
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



# Print:
plt.figure()
sns.set_style("whitegrid", {'axes.grid' : False})
plot_confusion_matrix(CM, classes=class_names, normalize=False,
                      title='Confusion matrix')
plt.show()


#=================================================================
#                  K-FOLD CROSS VALIDATION:
#=================================================================
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

# Define ANN classifier building function:
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size=10, epochs=10)
accuracies = cross_val_score(estimator = classifier, X=X_train, y=y_train, cv = 10)
#accuracies = cross_val_score(estimator = classifier, X=X_train, y=y_train, cv = 10, n_jobs=-1)
mean_acc = accuracies.mean()
variance_acc = accuracies.std()

#=================================================================
#                      GRID SEARCH TUNING: 
#=================================================================
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

# Define ANN classifier building function:
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)


# Define parameters to tune:
parameters = {'batch_size': [25, 32],
              'epochs': [10, 50], 
              'optimizer': ['adam', 'rmsprop']} 

# Create GridSearch instance:
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

# Fit to training set:
grid_search = grid_search.fit(X_train, y_train)

# Return best parameters:
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_



# PRINT RUN-TIME:
print("--- %s seconds ---" % (time.time() - start_time))















