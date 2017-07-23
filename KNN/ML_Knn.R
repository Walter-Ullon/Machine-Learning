# classification of cancer data using KNN Machine learnign algorithms

#import dataset
wbcd <- read.csv(url("https://raw.githubusercontent.com/dataspelunking/MLwR/master/Machine%20Learning%20with%20R%20(2nd%20Ed.)/Chapter%2003/wisc_bc_data.csv"))
#dataset info
str(wbcd)

#remove the unique "ID" feature for the observations (not a meaningful feature)
wbcd <- wbcd[-1]
table(wbcd$diagnosis)
wbcd$diagnosis <- factor(wbcd$diagnosis, levels = c("B", "M"), 
                         labels = c("Benign", "Malignant"))
round(prop.table(table(wbcd$diagnosis))*100, digits = 1)
summary(wbcd[c("radius_mean", "area_mean", "smoothness_mean")])

#min-max normalization function
normalize <- function(x){
  return ((x - min(x)) / (max(x) - min(x)))
}

#Normalizations using min-max and Z-scores (omit 1st column i.e. "ID")
wbcd_n <- as.data.frame(lapply(wbcd[2:31], normalize))
wbcd_z <- as.data.frame(scale(wbcd[-1]))

#confirm normalization is working as intended
summary(wbcd_z$area_mean)

#segmentation of dataset into Training & Test data (for both n and z norms)
wbcd_train <- wbcd_n[1:469, ]
wbcd_test <- wbcd_n[470:569, ]

wbcd_train_z <- wbcd_z[1:469, ]
wbcd_test_z <- wbcd_z[470:569, ]

#extraction of true labels for training and test results
wbcd_train_labels <- wbcd[1:469, 1]
wbcd_test_labels <- wbcd[470:569, 1]

#run the KNN algo with k = 21 (sqrt(469)) with normalized data
wbcd_test_pred <- knn(train = wbcd_train, test = wbcd_test, cl = wbcd_train_labels, k = 21)

#run the KNN algo with k = 21 (sqrt(469)) with z-normalized data
wbcd_test_pred_z <- knn(train = wbcd_train_z, test = wbcd_test_z, cl = wbcd_train_labels, k = 21)

#compare results for both n and z normalized data
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred, prop.chisq = FALSE)
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred_z, prop.chisq = FALSE)

