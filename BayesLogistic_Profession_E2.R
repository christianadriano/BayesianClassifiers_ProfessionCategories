"
Bayesian Logistic Regression to predict profession of participant in E2.

This model will be later used to infer the profession of participants in experiment 1.
Inferring professions in E1 will be later important to evaluate how well the causal models 
generalize across the two experiments E1 and E2.

We will compare different models based on different predictors:
gender
age
years of programming
qualification score
adjusted qualification score

"

library(dplyr)

#Load consent data from E2
source("C://Users//Christian//Documents//GitHub//CausalModel_FaultUnderstanding//data_loaders//load_consent_create_indexes_E2.R")
