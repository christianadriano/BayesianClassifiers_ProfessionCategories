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
library(rethinking)
library(dplyr)

#Load consent data from E2
source("C://Users//Christian//Documents//GitHub//CausalModel_FaultUnderstanding//data_loaders//load_consent_create_indexes_E2.R")

#create column with student and non-student
df_consent$is_student = FALSE

df_consent$profession_str <- as.character(df_consent$profession)
df_consent[df_consent$profession_str=="Undergraduate_Student",]$is_student <- TRUE
df_consent[df_consent$profession_str=="Graduate_Student",]$is_student <- TRUE

#Approximate 

# approximate posterior
m13.1 <- ulam(
  alist(
    S ~ dbinom( N , p ) ,
    logit(p) <- a[tank] ,
    a[tank] ~ dnorm( a_bar , sigma ) ,
    a_bar ~ dnorm( 0 , 1.5 ) ,
    sigma ~ dexp( 1 )
  ), 
  data=dat , chains=4 , log_lik=TRUE )