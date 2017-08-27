#----------------- Import packages: ----------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


#----------------- Load data and explore: ----------------------
college = pd.read_csv('College_Data', index_col = 0)

# Check stats for each category:
print()
print(college.describe())


#----------------- Visualize data: ----------------------

# Grad.Rate vs. Room.Board (by 'Private' column):
sns.set_style('whitegrid')
sns.lmplot(x='Grad.Rate', y='Room.Board', data=college, hue='Private', 
           palette='coolwarm',size=6, aspect=1, fit_reg=False)
plt.show()

# F.Undergrad vs. Outstate (by 'Private' column):
sns.lmplot(x='F.Undergrad', y='Outstate', data=college, hue='Private', 
           palette='coolwarm',size=6, aspect=1, fit_reg=False)
plt.show()


# Stacked histogram of out of state tuiton based on Private:
sns.set_style('darkgrid')
g = sns.FacetGrid(data=college,hue="Private",palette='coolwarm',size=6,aspect=2)
g = (g.map(plt.hist,'Outstate',bins=20,alpha=0.7)).add_legend()
plt.show()


# Stacked histogram of Grad.Rate based on Private:
college['Grad.Rate']['Cazenovia College'] = 100 # reset this school's grad rate to 100 (was above 100)

sns.set_style('darkgrid')
g = sns.FacetGrid(data=college,hue="Private",palette='coolwarm',size=6,aspect=2)
g = (g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)).add_legend()
plt.show()


#----------------- Prepare data for fitting: ----------------------
# Drop target category ("Private"):
X = college.drop('Private', axis =1)

#----------------- Train model: ----------------------
from sklearn.cluster import KMeans

# Create model instance. Set clusters to 2 (we assume there are 2 clusters to seggregate):
kmeans = KMeans(n_clusters=2)

# Fit:
kmeans.fit(X)

# Print cluster centers:
#print(kmeans.cluster_centers_)


#----------------- Evaluate model: ----------------------
'''
NOTE: There is no perfect way to evaluate clustering if we don't have the labels, 
however since this is just an exercise, we do have the labels, so we take advantage
of this to evaluate our clusters. We won't usually have this luxury in the real world. 
'''
# Create a new column for df called 'Cluster', which is a 1 for a Private school, and a 0 for a public school.
# define cluster classification function:
def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0
    
college['Cluster'] = college['Private'].apply(converter)

# Create a confusion matrix and classification report to see how well the Kmeans clustering worked without being given any labels.
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(college['Cluster'],kmeans.labels_))
print(classification_report(college['Cluster'],kmeans.labels_))


















