"
XGBoostTree to predict professions (Student and Non-Student) for E2 participants

Uses the Extreme Gradient Boosting, which is an efficient implementation 
of the gradient boosting frame-work from Chen & Guestrin (2016) <doi:10.1145/2939672.2939785>

Results:

1- Training setup
I reported both training and test error using 10-fold cross-validation. 
To mitigate overfitting of the decision trees, I set hyperparameters accordingly, 
for instance, depth=4 (intsead of standard 6), eta =0.3 (standard, lower the better),
stopping criteria to half of the rounds.

2- Feature selection
I selected the following features available after right after the qualification test:
years_programming, age, qualification_score, adjusted_score, test_duration

3- Results (by feature combination and sorted in descending order of test error)

TESTING ERRORS
#Considered Students = (Undergrads and Grads, only Undergrads)

#Only exogenous variables
age = (0.283,0.1934)
age, years_programming = (0.2768,0.2034)
age, years_programming,qualification_score = (0.2791,0.2014)

#Combined with Endogenous variables
age, years_programming,qualification_score,test_duration = (0.2645, 0.1921)
age, years_programming,adjusted_score,test_duration = (0.2629,0.1907)
age, years_programming,adjusted_score,test_duration,testDuration_fastMembership = (0.226,0.1615)


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

df_consent <- df_consent[!df_consent$profession_str == "Graduate_Student",]

df_consent$profession_str <- as.character(df_consent$profession)
df_consent[df_consent$profession_str %in% c("Undergraduate_Student","Graduate_Student"),]$is_student <- 1
df_consent[df_consent$profession_str %in% c("Professional_Developer", "Hobbyist", "Other","Programmer"),]$is_student <- 0

#Move all adjusted score to positive scale
df_consent$adjusted_score <- df_consent$adjusted_score + (-1*min(df_consent$adjusted_score)) + 0.1

train.features <- df_consent %>% select(years_programming,age,qualification_score)
train.label <- df_consent %>% select(is_student)

#-----------------------------

runXGBoost <- function(train.features,train.label,rounds,depth){
  bst <- xgboost(data = as.matrix(train.features), label = as.matrix(train.label),
                 max_depth =depth,eta = 1, nthread = 4, 
                 nrounds = rounds,eval_metric = list("error"), 
                 objective = "binary:logistic",verbose = 1);
  
  return (bst$evaluation_log[rounds-1][[2]])
}

runXGB_CrossValidation <- function(train.features,train.label){
  dtrain <- xgb.DMatrix(data = as.matrix(train.features), 
                        label = as.matrix(train.label));
  cv <- xgb.cv(data = dtrain, 
               nrounds = 1000, #larger rounds did not give better results
               early_stopping_rounds=500, #stop after error did not change and pick the best iteration
               nfold = 10, #best practice, smaller value, 5 fold gave similar results
               metrics = list("error"), #standard for binary classification
               max_depth = 4, #standard parameterization is 6, we reduced to 4 to reduce overfitting
               eta = 0.3, #standard, the smaller, the less overfitting risk
               objective = "binary:logistic",
               verbose=FALSE
               );
  
  print(paste0("train_error= ",round(cv$evaluation_log$train_error_mean[cv$best_iteration],digits=4)));
  print(paste0("test_error= ",round(cv$evaluation_log$test_error_mean[cv$best_iteration],digits=4)));
  
}

#-------------------------------------------
runXGB_CrossValidation(
  train.features = df_consent %>% select(age),
  train.label = train.label 
)

#[1] "train_error= 0.2786"
#[1] "test_error= 0.283"
# Only Undergrads as students
#[1] "train_error= 0.1934"
#[1] "test_error= 0.1934"

#-------------------------------------------
runXGB_CrossValidation(
  train.features = df_consent %>% select(age,years_programming),
  train.label = train.label 
)

#[1] "train_error= 0.2547"
#[1] "test_error= 0.2768"
# Only Undergrads as students
#[1] "train_error= 0.188"
#[1] "test_error= 0.2034"

#-------------------------------------------
runXGB_CrossValidation(
  train.features = df_consent %>% select(age,years_programming,qualification_score),
  train.label = train.label
  )

#[1] "train_error= 0.2498"
#[1] "test_error= 0.2791"
# Only Undergrads as students
#[1] "train_error= 0.1884"
#[1] "test_error= 0.2014"

#-------------------------------------------
runXGB_CrossValidation(
  train.features = df_consent %>% select(age,years_programming,qualification_score,test_duration),
  train.label = train.label
)

#[1] "train_error= 0.2245"
#[1] "test_error= 0.2645"
# Only Undergrads as students
#[1] "train_error= 0.1705"
#[1] "test_error= 0.1921"


#-------------------------------------------
runXGB_CrossValidation(
  train.features = df_consent %>% select(age,years_programming,adjusted_score,test_duration),
  train.label = train.label 
  )

#[1] "train_error= 0.2152"
#[1] "test_error= 0.2629"
# Only Undergrads as students
#[1] "train_error= 0.1663"
#[1] "test_error= 0.1907"

#-------------------------------------------
runXGB_CrossValidation(
  train.features = df_consent %>% select(age,years_programming,adjusted_score,test_duration,testDuration_fastMembership),
  train.label = train.label 
)

#[1] "train_error= 0.1836"
#[1] "test_error= 0.226"
# Only Undergrads as students
#[1] "train_error= 0.0103"
#[1] "test_error= 0.1615"

