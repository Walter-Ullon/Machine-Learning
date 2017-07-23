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
sms_dtm_test <- sms_dtm[4169:5559, ]

# create training and tests labels (from raw corpus)
sms_train_labels <- sms_raw[1:4169, ]$type
sms_test_labels <- sms_raw[4169:5559, ]$type

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



