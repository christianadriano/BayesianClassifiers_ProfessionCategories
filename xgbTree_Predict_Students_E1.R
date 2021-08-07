"
Classification Tree Learner to predict professions (Student and Non-Student) for E2 participants

- Builds a model using E2 data 
- Apply this model to predict which participants in E1 are probably students (undergraduate)

https://mlr3.mlr-org.com/reference/mlr_learners_classif.rpart.html

Results:

1- Training setup
I reported both training and test error using 10-fold cross-validation. 

2- Feature selection
I selected the following features available before the qualification test:
years_programming, age

3- Results (by feature combination and sorted in descending order of test error)

TESTING ERRORS
#Considered Students = (Undergrads and Grads,only Undergrads)

age =()
year_programming = ()
age, years_programming = (0.1868433,)

"

#install.packages("mlr3")
#install.packages("mlr3verse")
#install.packages("data.table")

library(data.table)
library(mlr3)
library(mlr3verse) #https://mlr3.mlr-org.com/
library(ggplot2)

library(xgboost)
library(dplyr)

#------------------------------------
#BUILD PREDICTION MODEL WIHT E2 DATA

#Load consent data from E2
source("C://Users//Christian//Documents//GitHub//CausalModel_FaultUnderstanding//data_loaders//load_consent_create_indexes_E2.R")
df_consent <- load_consent_create_indexes()

#create column with student and non-student
df_consent$is_student = 0
df_consent$is_student <-  factor(df_consent$is_student, levels = c(1,0))
df_consent$is_student <- as.factor(df_consent$is_student)

df_consent$profession_str <- as.character(df_consent$profession)
df_consent[df_consent$profession_str %in% c("Undergraduate_Student","Graduate_Student"),]$is_student <- 1
df_consent[df_consent$profession_str %in% c("Professional_Developer", "Hobbyist","Other","Programmer"),]$is_student <- 0

df_selected <- df_consent %>% select(years_programming,is_student)

#-----------------------------------------
#Cross-validation
#Used only age and years of programming as features

#Create TasK (mlr3 model)
task <- TaskClassif$new(df_selected, 
                        id = "training", 
                        target = "is_student")
#print(task)  
resampling = rsmp("cv", folds = 10)
resampling$instantiate(task)
resampling$iters
rr = resample(task, learner, resampling, store_models = TRUE)
#print(rr)

# average performance across all resampling iterations
rr$aggregate(msr("classif.ce")) 
#Consider as students only Undergrads
#> 0.1868433 (age, years_programing) <<<<BEST
#> 0.187358 (age, only undergrads)
#> 0.2477591 (years_programing, only undergrads)

#Consider as students Undergrads and Grads
#> 0.2891375 (age, years_programing) <<<<BEST
#> 0.2869311 (age)
#>  0.3517827 (years_programing)

#performance for the individual resampling iterations:
rr$score(msr("classif.ce"))
#  1:            cv         1 <PredictionClassif[19]>  0.1620112
#  2:            cv         2 <PredictionClassif[19]>  0.1061453
#  3:            cv         3 <PredictionClassif[19]>  0.1508380
#  4:            cv         4 <PredictionClassif[19]>  0.2178771
#  5:            cv         5 <PredictionClassif[19]>  0.2234637
#  6:            cv         6 <PredictionClassif[19]>  0.1284916
#  7:            cv         7 <PredictionClassif[19]>  0.1899441
#  8:            cv         8 <PredictionClassif[19]>  0.2011173
#  9:            cv         9 <PredictionClassif[19]>  0.1685393
# 10:            cv        10 <PredictionClassif[19]>  0.2359551


df <- data.frame(rr$score(msr("classif.ce")))
#View(df)
#Best learner
bestLearner <- df[which.min(df$classif.ce),]
#Worst learner
worstLearner <- df[which.max(df$classif.ce),]
#Median learner (one with the median error)
sorted <- df[order(df$classif.ce),]
Row <- sorted[6,]

#How to decide which learner from which fold to use to make the predictions?
#Maybe the best, worse, and the one close to the mean?
#Instead, I will use Auto-Tuning to find the best model


#-------------------------------------------
#AUTO-TUNING

#Using Nested Resampling
#https://mlr3book.mlr-org.com/nested-resampling.html#nested-resampling

learner = lrn("classif.rpart")
search_space = ps(cp = p_dbl(lower = 0.001, upper = 0.1))
terminator = trm("evals", n_evals = 5)
tuner = tnr("grid_search", resolution = 10)
resampling = rsmp("cv")
measure = msr("classif.ce")
auto_tuner = AutoTuner$new(learner, resampling, measure, terminator, tuner, search_space)

#Using cross-validation in the outerloop
outer_resampling = rsmp("cv", folds = 10)

rr = resample(task, at, outer_resampling, store_models = TRUE)

extract_inner_tuning_results(rr)
#       cp learner_param_vals  x_domain classif.ce
# 1: 0.089          <list[2]> <list[1]>  0.1711555
# 2: 0.034          <list[2]> <list[1]>  0.1870798
# 3: 0.056          <list[2]> <list[1]>  0.1837045

#The results above are around the average values of 
#the ones produced using he 10-fold cross-validation

rr$score()
#       resampling_id iteration           prediction classif.ce
# 1:            cv         1 <PredictionClassif[19]>  0.1963087
# 2:            cv         2 <PredictionClassif[19]>  0.1610738
# 3:            cv         3 <PredictionClassif[19]>  0.1895973

#Larger error in the oute resampling sugges that
#the models with the optimized hyperparameters are overfitting the data.

#hence, use the aggregate value of the outer resampling
rr$aggregate()
#classif.ce  = 0.1823266 

#FINAL MODEL produced by the auto-tuner
model <- auto_tuner$train(task)

#PLOT RESULTS E2
prediction = model$predict(task)
ggplot2::autoplot(prediction)

#-------------------------------------------
# Apply model to label E1
#Load consent data from E1
source("C://Users//Christian//Documents//GitHub//CausalModel_FaultUnderstanding//data_loaders//load_consent_create_indexes_E1.R")
df_consent_E1 <- load_consent_create_indexes()

#create column with student and non-student
df_consent_E1$is_student = 0
df_consent_E1$is_student <-  factor(df_consent_E1$is_student,levels=c(1,0))
df_consent_E1$is_student <- as.factor(df_consent_E1$is_student)
df_selected_E1 <- df_consent_E1 %>% select(years_programming,age,is_student)
#Filter out workers who did not provide age or years of programming
df_selected_E1 <-  df_selected_E1[!is.na(df_selected_E1$age) & !is.na(df_selected_E1$years_programming),]


task_test <- TaskClassif$new(df_selected_E1, 
                        id = "test", 
                        target = "is_student")

prediction_E1 = model$predict(task_test, row_ids = c(1:task_test$nrow))

#PLOT RESULTS E1
ggplot2::autoplot(prediction_E1)

#MERGE BACK IS_Student prediction RESULTS.

df_features <- data_frame(task_test$data())
df_response <- data_frame(response)

results <- cbind(df_features,df_response)

dplyr::left_join(df_selected_E1,prediction_E1,by =c("age","years_programming"), copy=FALSE)

#---------------------------------------
# WITHOUT CROSS-VALIDATION

#TASK
#Creaete task (mlr3 model)
task <- TaskClassif$new(df_selected, 
                        id = "worker_id", 
                        target = "is_student")
#print(task)

#LEARNER (2 step procedure)
learner = lrn("classif.rpart", id = "rp", cp = 0.001)

#train
train_set = sample(task$nrow, 0.8 * task$nrow)
test_set = setdiff(seq_len(task$nrow), train_set)

learner$train(task, row_ids = train_set)
#print(learner$model)

#predict
prediction = learner$predict(task, row_ids = test_set)
#print(prediction)
prediction$confusion
#           truth
# response   0   1
#        0 252  66
#        1   4  36

#CHANGING PREDICT TYPE TO PROBABILITY
learner$predict_type = "prob"
# re-fit the model
learner$train(task, row_ids = train_set)

# rebuild prediction object
prediction = learner$predict(task, row_ids = test_set)
#head(as.data.table(prediction))

#-----------------------------------
#PLOT RESULTS
learner = lrn("classif.rpart", predict_type = "prob")
learner$train(task)
prediction = learner$predict(task)
ggplot2::autoplot(prediction)

