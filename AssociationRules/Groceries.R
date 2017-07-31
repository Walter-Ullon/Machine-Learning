# Application of "Association Rule" algorithms to "Market Basket" analysis.
# Data from Packt publishing website.

# Install and load packages.
install.packages("arules")
library(arules)
library(arulesViz)
library(datasets)

# Load data using a "Sparse Matrix". Examine.
groceries <- read.transactions(url("https://raw.githubusercontent.com/dataspelunking/MLwR/master/Machine%20Learning%20with%20R%20(2nd%20Ed.)/Chapter%2008/groceries.csv"), sep = ",")
summary(groceries)
inspect(groceries[1:5])

# Inspect the "support" level of the 1st five items.
itemFrequency(groceries[ , 1:5])

# Create histogram of the items with a minimin support level of 0.1.
pdf("itemSupportLvl.pdf")
itemFrequencyPlot(groceries, support = 0.1)
dev.off()

# Create histogram of the top 20 items.
pdf("itemTop20.pdf")
itemFrequencyPlot(groceries, topN = 20)
dev.off()

# Visualize the Sparse Matrix.
pdf("itemSparseMatrix.pdf")
image(groceries[1:100])
dev.off()

# Create association rules. Summarize.
groceryRules <- apriori(groceries, parameter = list(support = 0.006, confidence = 0.25, minlen = 2))
groceryRules
summary(groceryRules)

# Inspect rules.
inspect(groceryRules[1:3])

# Sort association rules to bring up the more illuminating examples. Sort by "lift".
inspect(sort(groceryRules, by = "lift")[1:5])

# Sort rules by querying specific items into subsets (i.e. berries).
berryRules <- subset(groceryRules, items %in% "berries")
inspect(berryRules)

# Save rules to .CSV file.
write(groceryRules, file = "groceryRules.csv", sep = ",", quote = TRUE, row.names = FALSE)

# Convert rules to dataframe.
groceryRules_df <- as(groceryRules, "data.frame")


plot(berryRules, method="graph")