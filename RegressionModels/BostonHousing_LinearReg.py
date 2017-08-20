# Linear Regression Model applied to Boston Housing Price data.
# Data from 1978, copy of UCI ML housing dataset: http://archive.ics.uci.edu/ml/datasets/Housing


# Import Packages:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
 
# Import data. Examine:
boston = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", 
                     delim_whitespace=True, header=None)

# Assign names to columns based on the "names" file given by UCI: 
# https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names
attributes = ['crimeRate', 'landZoned lots', 'non-retail', 'tracksRiver', 'nitricOxide conc.', 
              'avgRooms', 'age', 'distToHubs', 'hwyAccess', 'tax', 'teachr/stud', 'propAfrAmer', 
              'lowStatus', 'medianVal']
boston.columns = attributes

# Examine: 
print(boston.head(10)) # return first 10 items.
print()
boston.info()
print()

# Statitics per Attribute
print(boston.describe())
print()

# Plot to check out the relationships between attributes.
#sns.pairplot(boston)
#plt.show()

# Show distribution of median values per thousand.
sns.distplot(boston['medianVal'])
plt.show()

# Heatmap of the correlation between attributes.
fig1 = plt.figure(figsize=(14, 10))
sns.heatmap(boston.corr(), annot=True, linewidths=.1)
plt.title("Attributes Correlation")
plt.show()
fig1.savefig("CorrHeatMap.pdf")

#----------------- Model training -------------------------
# Define explanatory (x) and response variables (y).
X = boston[['crimeRate', 'landZoned lots', 'non-retail', 'tracksRiver', 'nitricOxide conc.', 
              'avgRooms', 'age', 'distToHubs', 'hwyAccess', 'tax', 'teachr/stud', 'propAfrAmer', 
              'lowStatus']]
y = boston['medianVal']

# Split the data into TRAIN and TEST sets:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=3)

# Choose MODEL: LinearRegression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

# Fit Model to data:
lm.fit(X_train, y_train)


#------------------ Gather model information ----------------------
# Intercept
print("The Intercept is: "
      + str(lm.intercept_))
print()

# Explanatory variable coefficients:
coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['coefficients'])
print(coeff_df)



#------------------ Make predictions on the 'Test' set ----------------------
predictions = lm.predict(X_test)



#---------------- Evaluate predictions ---------------------
# Scatter plot of actual vs predicted (points on a straight line is a good sign)
# Save the next two plots as pdf.
fig2 = plt.figure()
plt.scatter(y_test, predictions)
plt.xlabel("actual")
plt.ylabel("prediction")
plt.show()
fig2.savefig("ActVsPredCorr.pdf", bbox_inches='tight')

# Distibution plot of residuals.
# Note: by the plot we can see that the residuals are normally distributed,
# this is good indication that the correct model was chosen.
fig3 = plt.figure()
sns.distplot((y_test-predictions), bins=50)
plt.title("Distribution of Residuals")
plt.show()
fig3.savefig("residualsDist.pdf", bbox_inches='tight')

#-------------- Metrics ---------------------
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))






