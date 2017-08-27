
#------------------- Load packages: -----------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

#------------------- Load data: -----------------------
ads = pd.read_csv('advertising.csv')
ads.dropna()

#------------------- Explore data: -----------------------
print(ads.describe())

# Age hisotgram:
plt.figure(figsize=(12,8))
ads['Age'].hist(bins=30)
#sns.countplot(x='Age', data=ads)
plt.show()

# Jointplot Age vs. Area Income:
sns.jointplot(x='Age', y='Area Income', data=ads)
plt.show()

# Jointplot Age vs. Daily Time Spent on Site:
sns.jointplot(x='Age', y='Daily Time Spent on Site', data=ads, kind='hex')
plt.show()


# Jointplot Daily Time Spent on Site vs. Daily Internet Usage:
sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=ads)
plt.show()

'''
# Pairplot categorized by 'Clicked on Ad' feature:
sns.pairplot(data=ads, hue='Clicked on Ad', palette='bwr')
plt.show()
'''


#------------------- Prepare data: -----------------------
from sklearn.model_selection import train_test_split

# Set predictors and response variables:
X = ads[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ads['Clicked on Ad']

# Split data: 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


#------------------- Train model and Fit: -----------------------
from sklearn.linear_model import LogisticRegression

# Create model instance:
LR = LogisticRegression()

# Fit:
LR.fit(X_train, y_train)

#------------------- Predict and Evaluate: -----------------------
predictions = LR.predict(X_test)

# Create classification report:
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))

# Print the probabilities for each of the samples in the test:
print(pd.DataFrame(LR.predict_proba(X_test), columns=LR.classes_))


# Visualize confusiion Matrix:
CM = confusion_matrix(y_test, predictions)
class_names = ['Click', 'No Click']

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

    #print(cm)

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




