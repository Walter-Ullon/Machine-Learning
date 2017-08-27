#-------------------- Load Packages: ---------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display


#-------------------- Load data & display Images: ---------------------
# Display Images:
print('Iris Setosa:')
url1 = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
display(Image(url1,width=300, height=300))
print()

print('Iris Versicolor:')
url2 = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
display(Image(url2,width=300, height=300))
print()

print('Iris Virginica:')
url3 = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
display(Image(url3,width=300, height=300))
print()

# Load data: 
iris = sns.load_dataset('iris')


#-------------------- Explore data & visualize: ---------------------
# Create pairplot:
sns.pairplot(data=iris, hue='species', palette='dark')
plt.show()

# KDE plot sepal length vs. sepal width for setosa:
setosa = iris[iris['species']=='setosa']
sns.kdeplot(setosa['sepal_width'], setosa['sepal_length'], cmap='plasma', shade=True, shade_lowest=False)
plt.show()


#-------------------- Split data: ---------------------
from sklearn.model_selection import train_test_split

# Set predictors and target categories:
X = iris.drop('species',axis=1)
y = iris['species']

#Split: 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


#-------------------- Train Model: ---------------------
from sklearn.svm import SVC

# Create model instance:
model = SVC()

# Fit:
model.fit(X_train, y_train)


#-------------------- Predict and evaluate: ---------------------
predictions = model.predict(X_test)

# Evaluate:
from sklearn.metrics import classification_report,confusion_matrix

# Confusion Matrix:
print(confusion_matrix(y_test,predictions))

# Classification Report:
print(classification_report(y_test,predictions))



#-------------------- Improve model results with GridSearch: ---------------------
from sklearn.model_selection import GridSearchCV

# Define list of parameters for the grid:
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 

# Create GridSearch obeject and fit it to the training data:
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)

grid.fit(X_train, y_train)

#----------------- Gather new Predictions and evaluate: ------------------------
grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test,grid_predictions))

print(classification_report(y_test,grid_predictions))














