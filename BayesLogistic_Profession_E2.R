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

TODO: 
1- simulate the priors
2- plot the coefficients
3- compute the prediction error
4- compared models w.r.t. overfitting (LOOC and WAIC)
5- compare these results with the xgboostTree

"
library(rethinking)
library(dplyr)

#Load consent data from E2
source("C://Users//Christian//Documents//GitHub//CausalModel_FaultUnderstanding//data_loaders//load_consent_create_indexes_E2.R")

#create column with student and non-student
df_consent$is_student = 0

df_consent$profession_str <- as.character(df_consent$profession)
df_consent[df_consent$profession_str=="Undergraduate_Student",]$is_student <- 1
df_consent[df_consent$profession_str=="Graduate_Student",]$is_student <- 1

#Move all adjusted score to positive scale
df_consent$adjusted_score <- df_consent$adjusted_score + (-1*min(df_consent$adjusted_score)) + 1


#Approximate posterior
m1 <- quap(
  alist(
  is_student ~ dbinom( 1 , p ) ,
  logit(p) <-  ba*age + by*years_programming+by + bq*qualification_score +bq,
  ba ~ dnorm( 1 , 10 ),
  by ~ dnorm( 1 , 10 ),
  bq ~ dnorm( 1 , 10 )
) , data=df_consent )
precis(m1)
#     mean   sd  5.5% 94.5%
# ba -0.01 0.00 -0.02  0.00
# by -0.12 0.01 -0.14 -0.10 OK
# bq  0.14 0.03  0.09  0.18 OK

m2 <- quap(
  alist(
    is_student ~ dbinom( 1 , p ) ,
    logit(p) <-  ba*age + by*years_programming + bq*adjusted_score +bq +1,
    ba ~ dnorm( 10, 10 ),
    by ~ dnorm( 1 , 10 ),
    bq ~ dnorm( 1 , 10) #CHOOSE THIS PRIOR APPROPRIATELY
  ) , data=df_consent )
precis(m2)

#     mean   sd  5.5% 94.5%
# ba -0.02 0.00 -0.03 -0.02 OK
# by -0.14 0.01 -0.15 -0.12 OK
# bq  0.42 0.06  0.14  0.37 OK

# m13.1 <- ulam(
#   alist(
#     is_student ~ dbinom( N , p ) ,
#     logit(p) <-,
#     age 
#     a[tank] ~ dnorm( a_bar , sigma ) ,
#     a_bar ~ dnorm( 0 , 1.5 ) ,
#     sigma ~ dexp( 1 )
#   ), 
#   data=dat , chains=4 , log_lik=TRUE )