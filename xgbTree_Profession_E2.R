"
XGBoostTree to predict professions (Student and Non-Student) for E2 participants

Uses the Extreme Gradient Boosting, which is an efficient implementation 
of the gradient boosting frame-work from Chen & Guestrin (2016) <doi:10.1145/2939672.2939785>

"

install.packages("xgboost")
library(xgboost)
library(dplyr)

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

train.features <- df_consent %>% select(years_programming,age,qualification_score)
train.label <- df_consent %>% select(is_student)

#-----------------------------

runXGBoost <- function(train.features,train.label,rounds,depth){
  bst <- xgboost(data = as.matrix(train.features), label = as.matrix(train.label),
                 max_depth =depth,eta = 1, nthread = 2, 
                 nrounds = rounds,eval_metric = list("rmse","auc"), 
                 objective = "binary:logistic",verbose = 1);
  
  return (bst$evaluation_log[rounds-1][[2]])
}

runXGB_CrossValidation <- function(train.features,train.label,rounds){
  dtrain <- xgb.DMatrix(data = as.matrix(train.features), label = as.matrix(train.label))
  cv <- xgb.cv(data = dtrain, nrounds = rounds, nthread = 2, nfold = 5, metrics = list("rmse","auc"),
               max_depth = 4, eta = 1, objective = "binary:logistic")
  return(cv)
}

#-----------------------------

#train-error:0.131432 

train_error <- runXGBoost(
  train.features = df_consent %>% select(years_programming),
  train.label = train.label ,
  rounds=1000,
  depth=4)
print(train_error)
#[1] 0.35179

cv_error <- runXGB_CrossValidation(
  train.features = df_consent %>% select(years_programming),
  train.label = train.label ,
  rounds=1000)

print(cv)
print(cv, verbose=TRUE)

#-------------------------------------------

train_error <- runXGBoost(
  train.features = df_consent %>% select(years_programming,age),
  train.label = train.label ,
  rounds=1000)
print(train_error)
#[1] 0.21085

train_error <- runXGBoost(
  train.features = df_consent %>% select(years_programming,age,qualification_score),
  train.label = train.label ,
  rounds=1000)
print(train_error)
#[1] 0.131432

train_error <- runXGBoost(
  train.features = df_consent %>% select(years_programming,age,adjusted_score),
  train.label = train.label ,
  rounds=1000)
print(train_error)
#[1] 0.0783
