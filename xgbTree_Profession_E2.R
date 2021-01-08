"
XGBoostTree to predict professions (Student and Non-Student) for E2 participants

Uses the Extreme Gradient Boosting, which is an efficient implementation 
of the gradient boosting frame-work from Chen & Guestrin (2016) <doi:10.1145/2939672.2939785>

"

install.packages("xgboost")
library(xgboost)

#------------------------------
#PRE-PROCESSING

#Load consent data from E2
source("C://Users//Christian//Documents//GitHub//CausalModel_FaultUnderstanding//data_loaders//load_consent_create_indexes_E2.R")

#create column with student and non-student
df_consent$is_student = 0

df_consent$profession_str <- as.character(df_consent$profession)
df_consent[df_consent$profession_str=="Undergraduate_Student",]$is_student <- 1
df_consent[df_consent$profession_str=="Graduate_Student",]$is_student <- 1

#Move all adjusted score to positive scale
df_consent$adjusted_score <- df_consent$adjusted_score + (-1*min(df_consent$adjusted_score)) + 0.1

#-----------------------------

bst <- xgboost(data = agaricus.train$data, label = agaricus.train$label, 
               max_depth = 2,eta = 1, nthread = 2, nrounds = 2, 
               objective = "binary:logistic")
