#=================================================================
#                      LOAD PACKAGES:
#=================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.misc import imread

#=================================================================
#                           LOAD DATA:
#=================================================================
housing = pd.read_csv('CaliforniaHousing1990.csv')

# Check first 5 values:
print(housing.head())
print()

# Print categories:
print(housing.columns.values)

# Print general info about the dataset. Notice missing values in categories:
print(housing.info())
print()

# Obtain general statistics on each category:
print(housing.describe())

#=================================================================
#                       VISUALIZE DATA:
#=================================================================
# Plot histogram for each category:
housing.hist(bins=50, figsize=(12,8))
plt.show()

# Jointplot median income vs. median house value:
sns.jointplot(x='median_income', y='median_house_value', data=housing, kind='hex')
plt.show()

# Plot geographical data:
# alpha -> blending value (0 = transparent)
# s -> marker radius
# c -> color sequence (like 'hue')

plt.figure(figsize=(14,9))
img=imread('California.png')

plt.imshow(img,zorder=0,extent=[housing['longitude'].min(),housing['longitude'].max(),housing['latitude'].min(),housing['latitude'].max()])
ax = plt.gca()
housing.plot(x='longitude', y='latitude', kind='scatter', alpha=0.4, 
         s= housing['population']/100, label='population', ax=ax,
         c= 'median_house_value', cmap=plt.get_cmap('jet'), colorbar=True, 
         zorder=5)
plt.legend()
plt.show()

#=================================================================
#                DEAL WITH MISSING VALUES:
#=================================================================
'''We will deal with messing values by replacing them with the median '''
# Check for null values:
print(housing[housing.isnull().any(axis=1)])

median = housing['total_bedrooms'].median()
housing['total_bedrooms'].fillna(median, inplace=True)

#=================================================================
#                      ENGINEER NEW FEATURES:
#=================================================================
# Create new feature to indicate # of rooms per household:
housing['rooms_per_household'] = housing['total_rooms'] / housing['households']

# Create new feature to indicate # of bedrooms per household:
housing['bedrooms_per_household'] = housing['total_bedrooms'] / housing['total_rooms']

# Create new feature to indicate # of occupants per household:
housing['population_per_household'] = housing['population'] / housing['households']


# Create a new categorical variable for 'median_income' (take median_income, divide by 1.5, and round up).
# Take any categories above 5, and set it as 5.
# Check new category histogram.
'''
In essence, we use income_cat as a way to split data in order to ensure that all train/test splits
contain an equal amount of representative samples.
Per expert's opinion, median income is a very important attribute to predict housing prices.
Thus, we create our 'strata' following this recommendation.
'''
housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True) # return values less than 5, else, return 5.
housing['income_cat'].hist(bins=10)
plt.show()

#=================================================================
#                 LOOK FOR CORRELATIONS IN DATA:
#=================================================================
# Populate correlation matrix:
corr_matrix = housing.corr()

# Check how much each attribute correlates with median_house_values:
from pandas.tools.plotting import scatter_matrix
print(corr_matrix['median_house_value'].sort_values(ascending=False))

# Create a subset of attributes and plot correlation matrix:
attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize=(12,8))
plt.show()

#=================================================================
#                HANDLE TEXT & CATEGORICAL DATA:
#=================================================================
from sklearn.preprocessing import LabelBinarizer

# Create encoder instance:
encoder = LabelBinarizer()

# Convert 'ocean_proximity' to a one-hot encoded vector: 
housing_cat_1hot = encoder.fit_transform(housing['ocean_proximity'])

# Turn 'one-hot' array to dataframe:
housing_cat_1hot = pd.DataFrame(housing_cat_1hot).reset_index(drop=True)

# Drop 'ocean_proximity' from dataframe:
housing.drop('ocean_proximity', axis = 1, inplace=True)

# concatenate 'housing' with 'one-hot' encoded categories:
housing = pd.concat([housing, housing_cat_1hot], axis=1).reset_index(drop=True)

#=================================================================
#                            NORMALIZE:
#=================================================================
from sklearn.preprocessing import StandardScaler

# Create scaler instance:
scaler = StandardScaler()

# Apply:
scaler.fit(housing)

#=================================================================
#                      PREPARE DATA & SPLIT:
#=================================================================
# Split data into train and test sets using stratified sampling based on 'income_cat':
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in sss.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Remove 'income_cat' to return data to original state:
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)

# Set training set: make X_train contain everything but target variable.
X_train = strat_train_set.ix[:, strat_train_set.columns != 'median_house_value']
y_train = strat_train_set['median_house_value']

X_test = strat_test_set.ix[:, strat_test_set.columns != 'median_house_value']
y_test = strat_test_set['median_house_value']
#=================================================================
#                   TRAIN MODEL & PREDICT:
#=================================================================
from sklearn.linear_model import LinearRegression

# Create model instance:
lr = LinearRegression()
lr.fit(X_train, y_train)

predictions = lr.predict(X_test)

#===============================================================
#                         PRINT ERROR:
#===============================================================
from sklearn.metrics import mean_squared_error

lin_MSE = mean_squared_error(y_test, predictions)
print(lin_MSE)

print(predictions[:5])
print(y_test[:5])

# Results summary function:
def results_df(truth, pred, num_results):
    s = pd.DataFrame(truth).reset_index(drop=True)
    r = pd.DataFrame(pred).reset_index(drop=True)
    t = pd.concat([s, r], axis=1).reset_index(drop=True)
    
    t.columns = ['truth', 'prediction']
    t['difference'] = t['truth'] - t['prediction']
    t['%'] = abs(t['difference'] / t['truth'])

    print(t.head(num_results))
    
    
# Print Results:
results_df(y_test, predictions, 20)

#===============================================================
#                   PRINT MODEL COEFFICIENTS:
#===============================================================
coeffs = pd.DataFrame(list(zip(X_train.columns, lr.coef_)), columns=['features', 'coeffs'])
print(coeffs)


#===============================================================
#                   CROSS VALIDATION:
#===============================================================
from sklearn.model_selection import cross_val_score

# Ten folds:
lr_scores = cross_val_score(lr, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
lr_rmse_scores = np.sqrt(-lr_scores)

# Score-printing function:
def display_scores(scores):
    print('Scores: ', scores)
    print('Mean: ', scores.mean())
    print('Standard Deviation: ', scores.std())

# Print scores:
display_scores(lr_rmse_scores)


#===============================================================
#         TRAIN DIFFERENT MODEL: RANDOM FOREST REGRESSOR
#===============================================================
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_predictions = rfr.predict(X_train)


# Evaluate results:
results_df(y_test, rfr_predictions, 20)

#===============================================================
#                       GRID SEARCH:
#===============================================================
from sklearn.model_selection import GridSearchCV

# Set Parameters to iterate through:
param_grid = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
              {'bootstrap': [False], 'n_estimators': [3, 10, 30], 'max_features': [2, 3, 4, 6, 8]}]

# Create Grid Search instance: 5 folds.
grid_search = GridSearchCV(rfr, param_grid, cv=5, scoring='neg_mean_squared_error')

# Apply search:
grid_search.fit(X_train, y_train)

# Print best parameters result:
print(grid_search.best_params_)
print(grid_search.best_estimator_)



#===============================================================
#                  GET FEATURE IMPORTANCE:
#===============================================================
# Get numerical scores for feature importance:
feature_importance = grid_search.best_estimator_.feature_importances_

#extra_attribs = ['rooms_per_hhold', 'pop_per_hhold', 'bedrooms_per_room']

# Get column names for one-hot encoded categories:
cat_one_hot_attribs = list(encoder.classes_)

# Get column names for rest of data:
num_attribs = list(X_train)
attributes = num_attribs

# Print features along with their scores:
vals = sorted(zip(feature_importance, attributes), key=lambda x: x[0], reverse=True)
df = pd.DataFrame(vals)
df.iloc[:, -1] = df.iloc[:, -1].replace({i : k for i, k in enumerate(cat_one_hot_attribs)})
print(df)







