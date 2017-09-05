# Logistic Regression algorithm applied to the Titanic Dataset from Kaggle.

# Import packages:
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

# load data
train = pd.read_csv("titanic_train.csv")
test = pd.read_csv("titanic_test.csv")

print(train.head())

#------------------- Inspect the data --------------------
# Use a heatmap to visualize missing values:
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Missing Values - Train')
plt.show()

sns.heatmap(test.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Missing Values - Test')
plt.show()

# Visualize proportion of Passengers who survived vs Drowned, separated by Sex:
sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train, hue='Sex', palette='RdBu_r')
plt.title("Survival vs. Sex")
plt.show()

# Visualize proportion of Passengers who survived vs Drowned, separated by PassengerClass:
sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train, hue='Pclass')
plt.title("Survival vs. Passenger Class")
plt.show()

# Visualize survival age distribution:
sns.distplot(train['Age'].dropna(), bins=30)
plt.title("Age Distribution of Passengers")
plt.show()

# Visualize proportion of Passengers who had children, siblings, spouses:
sns.countplot(x='SibSp', data=train)
plt.title('Proportion of Passengers with Siblings, Children, Spouses.')
plt.show()

# Visualize distribution of fares paid by passengers:
sns.distplot(train['Fare'], kde=False, bins=40)
plt.show()

#------------------ Engineer new Features --------------------
# get the "title" prefix before each passenger's name.
train['Name'] = train['Name'].apply(lambda x: x.replace('.',','))
test['Name'] = test['Name'].apply(lambda x: x.replace('.',','))

train['Prefix'] = train['Name'].apply(lambda x: x.split(',')[1])
test['Prefix'] = test['Name'].apply(lambda x: x.split(',')[1])

print(train['Prefix'].value_counts())

# Function sorts prefix to reduce dimension
def prefix_sort(prefix):
    rare = [' Dr', ' Rev', ' Col', ' Major', ' Don', ' Sir', ' the Countess',
            ' Jonkheer', ' Capt']
    misses = [' Lady', ' Ms', ' Mlle']
    missus = [' Mme', ' Dona', ' Lady']
    if prefix in rare:
        return 'rare'
    elif prefix in misses:
        return ' Miss'
    elif prefix in missus:
        return ' Mrs'
    elif prefix == ' Master':
        return ' Mr'
    else:
        return prefix
    
# Apply function to prefix column
train['Prefix'] = train['Prefix'].apply(prefix_sort)
test['Prefix'] = test['Prefix'].apply(prefix_sort)

print(train['Prefix'].value_counts())
print()
print(test['Prefix'].value_counts())
    
# We introduce a new category: Totalfamiliy (fam size)
# This is a result from adding siblings + parents + 1 (to account for the individual himself)      
train['totalFamily'] = train['SibSp'] + train['Parch'] + 1
test['totalFamily'] = test['SibSp'] + test['Parch'] + 1
print(train['totalFamily'].value_counts())
print(test.head())

# Turn family size into a category
def fam_size(fam):
    if fam <= 4:
        return "small"
    else:
        return 'large'
    
# Apply fam_size function to 'totalFamily'
train['FamSize'] = train['totalFamily'].apply(fam_size)
test['FamSize'] = test['totalFamily'].apply(fam_size)
print(train['FamSize'].value_counts())

#------------------ Clean up Data --------------------
# We will replace (impute) missing values by the mean age by passenger class.
# Visualize age distribution by passenger class.
sns.boxplot(x='Pclass', y='Age',data=train)
plt.title("Age distribution by passenger class - Train")
plt.show()

sns.boxplot(x='Pclass', y='Age',data=test)
plt.title("Age distribution by passenger class - Test")
plt.show()

# Get mean age per passenger class and print it to the screen.
AvgAgePclass_df = round(train.groupby('Pclass')['Age'].mean())
print('The mean age for passengers in each class is: ' + str(AvgAgePclass_df))

# Define function imputation function for age:
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return AvgAgePclass_df[1]
        elif Pclass == 2:
            return AvgAgePclass_df[2]
        else: 
            return AvgAgePclass_df[3]
        
    else:
            return Age
        
# Define imputation function for fare:
AvgFarePclass_df = round(train.groupby('Pclass')['Fare'].mean()) # get mean fare given Pclass
print('The mean fare for passengers in each class is: ' + str(AvgFarePclass_df))

def impute_fare(cols):
    fare = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(fare):
        
        if Pclass == 1:
            return AvgFarePclass_df[1]
        elif Pclass == 2:
            return AvgFarePclass_df[2]
        else: 
            return AvgFarePclass_df[3]
        
    else:
            return fare        

# Apply age imputation function to data:
# Note: we will impute the same mean values from the train set to the test set
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1) # axis=1 since we want to apply it to the columns
test['Age'] = test[['Age', 'Pclass']].apply(impute_age, axis=1) # axis=1 since we want to apply it to the columns

# Apply fare imputation function to data:
# Note: we will impute the same mean values from the train set to the test set
test['Fare'] = test[['Fare', 'Pclass']].apply(impute_fare, axis=1) # axis=1 since we want to apply it to the columns

# Re-check 'null' heatmap again:
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Missing Values - Train')
plt.show()

sns.heatmap(test.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Missing Values - Test')
plt.show()

sns.heatmap(test.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

# Since we have too mnay missing values in the 'cabin' column, we will drop it:
train.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)


# Drop last remaining records with mising values:
# print(test[test.isnull()==False].count()) #check for na values in test
train.dropna(inplace=True)
test.dropna(inplace=True)

# We need to convert Categorical variables into numerical ones.
# We accomplish this by creating "dummy' variables.
# We use the 'get_dummies()' method on the 'Sex','Embarked', 'Pclass' categories to return a numerical 
# representation of these columns. However, in order to avoid co-linearity problems,
# we drop the 1st column in the resulting dataframe. i.e 1 if male, otherwise 0
sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)
pclass = pd.get_dummies(train['Pclass'], drop_first=True)
prefix = pd.get_dummies(train['Prefix'], drop_first=True)
famSize = pd.get_dummies(train['FamSize'], drop_first=True)

# Concatenate the new columns into the train dataset:
train = pd.concat([train, sex, embark, pclass, prefix, famSize], axis=1)

# We drop the uncoded sex, embarked, columns, as well as any other columns that 
# will not provide us with usefulf information (name, ticket#, passengerID)
train.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'Pclass','PassengerId', 'Prefix',
            'totalFamily', 'FamSize', 'SibSp', 'Parch'], axis=1, inplace=True)

# do the same steps above on the 'test' set:
sex = pd.get_dummies(test['Sex'], drop_first=True)
embark = pd.get_dummies(test['Embarked'], drop_first=True)
pclass = pd.get_dummies(test['Pclass'], drop_first=True)
prefix = pd.get_dummies(test['Prefix'], drop_first=True)
famSize = pd.get_dummies(test['FamSize'], drop_first=True)

# Concatenate the new columns into the train dataset:
test = pd.concat([test, sex, embark, pclass, prefix, famSize], axis=1)

# We drop the uncoded sex, embarked, columns, as well as any other columns that 
# will not provide us with usefulf information (name, ticket#, passengerID)
test.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'Pclass', 'PassengerId', 'Prefix', 
           'totalFamily', 'FamSize', 'SibSp', 'Parch'], axis=1, inplace=True)


#------------------ Train Model --------------------
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 

X_train = train.drop('Survived', axis=1)
y_train = train['Survived']

#X_train = scaler.fit_transform(X_train)
#y_train = scaler.fit_transform(y_train)


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

# Predict:
predictions = logmodel.predict(test)

passId = np.arange(892,1310)
predictions = pd.DataFrame(predictions, columns=['Survived'] )

# prepare dataframe to save as CSV for submission.
predictionsCSV = pd.DataFrame(passId, columns=['PassengerId'] )
predictionsCSV['Survived'] = predictions

predictionsCSV.to_csv('TitanicPrediction.csv', index=False)

# Evaluate:
#from sklearn.metrics import classification_report
#print(classification_report(test, predictions)) #note: we do not have the actuals for this set (kaggle comp.)