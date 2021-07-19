"
XGBoostTree to predict professions (Student and Non-Student) for E2 participants

- Builds a model using E2 data 
- Apply this model to predict which participants in E1 are probably students (undergraduate)

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
age, years_programming,adjusted_score,test_duration,testDuration_fastMembership = 0.226


"

install.packages("mlr3")
install.packages("mlr3verse")
install.packages("data.table")

library(data.table)
library(mlr3)
library(mlr3verse)


library(xgboost)
library(dplyr)

#------------------------------------
#BUILD PREDICTION MODEL WIHT E2 DATA

#Load consent data from E2
source("C://Users//Christian//Documents//GitHub//CausalModel_FaultUnderstanding//data_loaders//load_consent_create_indexes_E2.R")
df_consent <- load_consent_create_indexes()

#create column with student and non-student
df_consent$is_student = 0

df_consent$profession_str <- as.character(df_consent$profession)
df_consent[df_consent$profession_str %in% c("Undergraduate_Student"),]$is_student <- 1
df_consent[df_consent$profession_str %in% c("Professional_Developer", "Hobbyist", "Graduate_Student","Other","Programmer"),]$is_student <- 0
df_consent$is_student <-  as.factor(df_consent$is_student)

#-----------------------------------------

#https://mlr3.mlr-org.com/
#TODO
#Implement cross-validation using xgboostTree
#use only age and years of programming as features

#Creaete task (mlr3 model)
task_students <- TaskClassif$new(df_consent, 
                                 id = "worker_id", 
                                 target = "is_student")
print(task_students)
#-------------------------------------------
model <- runXGB_CrossValidation(
  train.features = df_consent %>% select(age,years_programming),
  train.label = train.label 
)

#------------------------------
#PRE-PROCESSING

#Load consent data from E1
source("C://Users//Christian//Documents//GitHub//CausalModel_FaultUnderstanding//data_loaders//load_consent_create_indexes_E1.R")

#create column with student and non-student
df1_consent$is_student = 0

train.features <- df1_consent %>% select(years_programming,age,qualification_score)
train.label <- df1_consent %>% select(is_student)

#-----------------------------

