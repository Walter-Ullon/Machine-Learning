# Application of "K-Means Clustering" algortihms to Teen Market Segements using social media data.
# Data collected and curated by Brett Lantz (https://raw.githubusercontent.com/dataspelunking)

# Install and load packages.
library(stats)

# Load data. Inspect.
teens <- read.csv(url("https://raw.githubusercontent.com/dataspelunking/MLwR/master/Machine%20Learning%20with%20R%20(2nd%20Ed.)/Chapter%2009/snsdata.csv"))
str(teens)
table(teens$gender, useNA = "ifany") # We have 2724 missing points in the gender column.
summary(teens$age)

# Since the "age summary" presented us with problematic statistics, we will have to clean up the data.
teens$age <- ifelse(teens$age > 13 & teens$age < 20, teens$age, NA)
summary(teens$age)

# Use "imputation" to deal with the missing values in age. Use the mean age to impute said values.
# Specifically, we need the mean for each of the graduation years.
mean(teens$age, na.rm = TRUE)
aggregate(data = teens, age ~ gradyear, mean, na.rm = TRUE)
# Above: "in data = teens, sort age by gradyear, and apply mean to each. Remove NA values".
ave_age <- ave(teens$age, teens$gradyear, FUN = function(x) mean(x, na.rm = TRUE))
# We use an "ifelse" to replace the NA values with the new gradyear age means.
teens$age <- ifelse(is.na(teens$age), ave_age, teens$age)
# Check our work...
summary(teens$age)

# Create "dummy variables" in order to solve the missing values in "gender". Assign "NA" values to "no_gender".
teens$female <- ifelse(teens$gender == "F" & !is.na(teens$gender), 1, 0)
teens$no_gender <- ifelse(is.na(teens$gender), 1, 0)

# Create a data frame containing only the "interests" of this group (columns 5:40).
interests <- teens[5:40]

# Train Model on the data. Use k = 5 (we predict 5 "teen" types).
interests_z <- as.data.frame(lapply(interests, scale))
set.seed(2345)
teens_cluster <- kmeans(interests_z, 5)

# Evaluate Model Performance.
teens_cluster$size
teens_cluster$centers

#--------------- Improve Model Performance -------------------
# The kmeans() function includes a component named "cluster" that contains the cluster assignments
# for each of the 30,000 individuals in the sample. We will add this as a column to "teens" data.
teens$cluster <- teens_cluster$cluster

# Obatin mean proportion of females per cluster. 
aggregate(data = teens, female ~ cluster, mean)

# Obatin mean proportion of friends per cluster. 
aggregate(data = teens, friends ~ cluster, mean)



