"""
@author: walterullon
"""

# NLP algorithm applied to sentiment analysis in YELP reviews.
# Data from Kaggle.

#----------------- Import packages: ---------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk # Imports the library
# nltk.download() #Download the necessary datasets
from nltk.corpus import stopwords


#----------------- Load data and inspect: ---------------------
yelp = pd.read_csv('yelp.csv')
print('1*------------------------')
print(yelp.head())

print('2*------------------------')
yelp.info()

print('3*------------------------')
print(yelp.describe())


#----------------- Create new column. Visualize data: ---------------------
yelp['text length'] = yelp['text'].apply(len)
print('4*------------------------')
print(yelp.head())

print('5*------------------------')
sns.FacetGrid(yelp, col='stars').map(plt.hist, 'text length', bins=30)
plt.show()

print('6*------------------------')
sns.boxplot(x='stars', y='text length', data=yelp)
plt.show()

print('7*------------------------')
sns.countplot(x='stars', data=yelp)
plt.show()

print('8*------------------------')
print('Mean # of stars per numerical category')
starsMean = yelp.groupby('stars').mean()
print(starsMean)

print('9*------------------------')
print('Correlation between mean # of stars per numerical category')
print(starsMean.corr())

print('10*------------------------')
sns.heatmap(starsMean.corr(), annot=True)
plt.title('correlation heatmap')
plt.show()


#----------------- Prepare data for classification: ---------------------

# Define text-processing function:
def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]



# Create new dataframe consisting of only those reviews that received 1 or 5 stars.
yelp_class = yelp[(yelp['stars']==1) | (yelp['stars']==5)]

# Create two objects: the 'text' review, and the labels ('stars').
X = yelp_class['text']
y = yelp_class['stars']

# Import CountVectorizer() and apply to text.
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
print('11*------------------------')
print(X.head(10))
X = cv.fit_transform(X)
print()
print('Shape of document term Matrix:')
print(X.shape)

# Split data into training/test:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


#----------------- Train Model: ---------------------
# We will use a NaiveBayes classifier
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# Fit:
nb.fit(X_train, y_train)

# Predict:
predictions = nb.predict(X_test)


#----------------- Evaluate results: ---------------------
from sklearn.metrics import confusion_matrix,classification_report

# Print confusion matrix and classification report:
print('12*------------------------')
print(confusion_matrix(y_test, predictions))
print()
print(classification_report(y_test, predictions))




'''
Next, we seek to improve results by applying text processing to our data.
We will employ TF-IDF, 'bag-of-words', etc...
We will switch to a RandonForest() classifier as the NB did poorly on the pipeline.
'''

'''
#----------------- Apply Text Precessing Using a 'Pipeline' ---------------------
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

# Build the Pipeline
# Feed 'text_process' into pipeline.
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', RandomForestClassifier(n_estimators=300)),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

# Re-do the data split:
X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

# Run the data through the pipeline:
pipeline.fit(X_train,y_train)

# Predict using the pipeline's output:
predictions = pipeline.predict(X_test)


# Print confusion matrix and classification report:
print('13*------------------------')
print(confusion_matrix(y_test, predictions))
print()
print(classification_report(y_test, predictions))





'''


#businesses = pd.read_json('yelp_training_set/yelp_training_set_business.json',lines=True)
#users = pd.read_json('yelp_training_set/yelp_training_set_user.json',lines=True)
#checkIn = pd.read_json('yelp_training_set/yelp_training_set_checkin.json',lines=True)
#reviews = pd.read_json('yelp_training_set/yelp_training_set_review.json',lines=True)













