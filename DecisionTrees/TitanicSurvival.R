# Survival on the Titanic

# Load Data
titanic_train <- read.csv(url("https://storage.googleapis.com/kaggle-competitions-data/kaggle/3136/train.csv?GoogleAccessId=competitions-data@kaggle-161607.iam.gserviceaccount.com&Expires=1501202527&Signature=kH%2B5q%2F3a9R9v4LVII%2FBnOYCfO0In8FiJMD7RNYEHtWWnY1HGYnR7KOJx0UWVwDaVpWx2Hw6Ds8%2FPSKzO3V8DjkjDByuBmNQunOCPWfwh2RH79cNBHGQDhHhoR6ttdV8gDPJq5EGwWvHT%2BTZbGGwKEDLrIChKpnV90ScbT9DwXELp1a%2BhciCPV4RKasuTIo37%2FvFuJY4MFCWbPj%2B9eFUfHxyfF3FEKEJ%2FN7gYznyb%2BMkUnr3d3Kuh%2BXFtGO3zwNXDx0LZI5FUtxkqeym1xKAtAUSDsIv1OW%2FIoL4pH9zHoysSU0JLltZOtXBY74zouw1xm23lfcA25AJULGQ47aitRg%3D%3D"), stringsAsFactors = FALSE)
titanic_test <- read.csv(url("https://storage.googleapis.com/kaggle-competitions-data/kaggle/3136/test.csv?GoogleAccessId=competitions-data@kaggle-161607.iam.gserviceaccount.com&Expires=1501202573&Signature=hNq96PKu5916QwmTLsKox0afkVwsdRGWrsRlloxH5zwgXdJ2BCREDCOqX%2FsYraKszG7LkC%2FF1MFT%2FMJ1Ccb7vFmk6jlPCL9ic2L4S7PIBOBP29OsEhh8Wu0%2BXRbCjAwBvAzJVBEzSCDcIhemPLLTUXbh6hfXLOeKfFVbTwM6oqrUxGVQta6iJpnJmD96o2aOqsfdTtNNG8Hmxqhp15EDwWC7VcHOFxaZvwBrprWrbvxF82ILfDsjqPl5AnYzZhOCYu%2FF2VueKceTQ5eXTatKEM9%2B0D6K8XCIkswsehSJv8IrhYa%2BcoTiA%2FKsbcrnSLu7jdORwpHJBS6XaLDh3f8m5w%3D%3D"), stringsAsFactors = FALSE)

# Inspect data
str(titanic_test)
table(titanic_train$Sex)

# Remove unnecessary & unique variables
titanic_train <- titanic_train[c(-1, -4, -9, -10, -11, -12)]
titanic_test <- titanic_test[c(-1, -3, -8, -9, -10, -11)]

# Train model
titanic_model <- C5.0(titanic_train[-1], as.factor(titanic_train$Survived))
summary(titanic_model)
pdf("titanicSurvivalTree.pdf", width=16,height=7)
plot(titanic_model)
dev.off()

# Make prediction
titanic_pred <- predict(titanic_model, titanic_test)

# Obtain proportions
prop.table(table(titanic_test$Sex))
prop.table(table(titanic_pred))

# Save as .CSV and prep data for submission to Kaggle.
write.csv(titanic_pred, file = "titanicPrediction.csv")

          