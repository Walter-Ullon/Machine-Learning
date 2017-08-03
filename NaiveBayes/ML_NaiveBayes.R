# Spam classifier using a Naive Bayes Machine Learning Algorithm. 
# Data collected from http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/

# install packages
install.packages("tm")
install.packages("NLP")
install.packages("SnowballC")
install.packages("wordcloud")
install.packages("e1071")
install.packages("gmodels")
install.packages("caret")
install.packages("ROCR")
# load packages
library(tm)
library(NLP)
library(SnowballC)
library(wordcloud)
library(e1071)
library(gmodels)
library(caret)
library(ROCR)

# import data
sms_raw <- read.csv(url("https://raw.githubusercontent.com/dataspelunking/MLwR/master/Machine%20Learning%20with%20R%20(2nd%20Ed.)/Chapter%2004/sms_spam.csv"), stringsAsFactors = FALSE)

# turn categorical 'type' variable into factor
sms_raw$type <- factor(sms_raw$type)

# get info on imported data
str(sms_raw)
table(sms_raw$type)

# create corpus of SMS messages
sms_corpus <- VCorpus(VectorSource(sms_raw$text))
print(sms_corpus)
# check 1st element of corpus
as.character(sms_corpus[[1]])
# check a list of corpus elements
lapply(sms_corpus[1:3], as.character)

#------------begin cleaning corpus---------------------
# apply lower case transformation to corpus, and check...
sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))
lapply(sms_corpus_clean[1:3], as.character)

# remove numbers from corpus
sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers)

# remove stop words from corpus (to, and, but, ...)
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords())

# remove punctuation from corpus
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)

# stem corpus ( learned, learning, learns -> learn)
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)

# strip whitespace
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)
#--------------corpus cleaning finalized---------------------

# create a "Document Term Matrix" (DTM). Creates matrix word count per document (SMS).
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)

# create training and tests sets
sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test <- sms_dtm[4170:5559, ]

# create training and tests labels (from raw corpus)
sms_train_labels <- sms_raw[1:4169, ]$type
sms_test_labels <- sms_raw[4170:5559, ]$type

# compare proportion of ham vs spam in both sets
prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))

#create wordcloud to visulaize frequency of words
wordcloud(sms_corpus_clean, min.freq = 50, random.order = FALSE, random.color = FALSE)

# create and compare wordclouds for ham vs spam (wordcloud applies text prep. processes automatically)
spam <- subset(sms_raw, type == "spam")
ham <- subset(sms_raw, type == "ham")
wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))

# create a vector with the most frequent words (appearing at least five times)
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)
str(sms_freq_words)

# create training and test sets of frequent words
sms_dtm_freq_train <- sms_dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[ , sms_freq_words]

# convert numeric features to categorical (if word appears, "YES", else, "NO")
# This conversion is needed since the Naive Bayes classifier is trained on categorical data
convert_counts <- function(x) {
  ifelse(x > 0, "Yes", "No")
}

# create training and test matrices (MARGIN = 2 -- "apply to columns")
# save matrices to CSV file (rows = document #, colums = word instance (yes/no))
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2, convert_counts)
write.csv(sms_train, file = "sms_train.csv")
write.csv(sms_test, file = "sms_test.csv")

# Build SMS Classifier Model
sms_classifier <- naiveBayes(sms_train, sms_train_labels)

# Make predicitons using the classifier model on the test data store them in a vector.
# Run the prediction a second time and obtain the "confidence" in terms of probabilities.
sms_test_pred <- predict(sms_classifier, sms_test)
sms_test_prob <- predict(sms_classifier, sms_test, type = "raw")
head(sms_test_prob)

#------------ Present findings in a data table. ----------------
sms_results <- data.frame(sms_test_labels, sms_test_pred, sms_test_prob)
col_headings <- c('Actual type','Predicted type', 'Prob. Ham','Prob. Spam')
names(sms_results) <- col_headings


# Compare predicitons
CrossTable(sms_test_pred, sms_test_labels, prop.chisq = FALSE, prop.t = FALSE, dnn = c("predicted", "actual"))

# Improve the model by setting a Laplace Estimator of 1
sms_classifier2 <- naiveBayes(sms_train, sms_train_labels, laplace = 1)
sms_test_pred2 <- predict(sms_classifier2, sms_test)
sms_test_prob2 <- predict(sms_classifier2, sms_test, type = "raw")
CrossTable(sms_test_pred2, sms_test_labels, prop.chisq = FALSE, prop.t = FALSE, dnn = c("predicted", "actual"))

#------------ Present findings in a data table. V2 ----------------
sms_results2 <- data.frame(sms_test_labels, sms_test_pred2, sms_test_prob2)
col_headings2 <- c('Actual type','Predicted type', 'Prob. Ham','Prob. Spam')
names(sms_results2) <- col_headings2

#------------- Analyze model performance with 'Caret' ------------------
confusionMatrix(sms_test_pred2, sms_test_labels, positive = "spam")

#------------- Visualize model performance ------------------
# prediction needs two vectors: estimated spam probs. and actual class labels.
pred <- prediction(predictions = sms_test_prob2[ ,2], labels = sms_test_labels)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
pdf("SMS_ROCcurve.pdf")
plot(perf, main = "ROC curve for SMS spam filter", col = "blue", lwd = 3)
dev.off()

# Calculate area under the ROC Curve.
perf.auc <- performance(pred, measure = "auc")
unlist(perf.auc@y.values)




