
install.packages("C50")
library(C50)
library(gmodels)

# Import data from Kaggle (https://www.kaggle.com/uciml/mushroom-classification)
shrooms <- read.csv(url("https://storage.googleapis.com/kaggle-datasets/478/974/mushrooms.csv
                        ?GoogleAccessId=datasets@kaggle-161607.iam.gserviceaccount.com
                        &Expires=1501124301&Signature=jbRzz1BZ%2B0vuVCExQWpDZrinPfpx54HWrrWW
                        gSAlX64SLxugJ9c03TfmvxP7wSe0Miao5DutK5FiKDsdR4wMsjV16VObIoBnHYobAZHhE8j
                        TNFfn7%2FWjyQ3wnKsakiv0fK8zAwBuXw6r7C71DDmlPQi8CyQdtYdUUh4IQ893dnmfbF7t
                        FGlJd84UwTMQKciYHsGMz4L3vxDYFqygYwoKuKu7vB7btPaGRHbzIOXda96BYbyDOGumsIiwvgoh
                        OngO5Mpc1iO%2Bh2gepNImglISOgfdAzl7%2ByiOTCjCKp0T1jiYv2sOCsmBkJSMSjrMpxVJltxXRut
                        BHAgIr9lxN%2BPe3A%3D%3D"))
# Inspect data
str(shrooms)

# Distribution of edible vs poisonous shrooms
table(shrooms$class)

