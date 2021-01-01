# Bayesian Classifiers to Infer Profession Categories Across Experiments
 Classification of subjects professions using Bayesian Random Forest and Bayesian Logistic Regression

Goals: 
1- build models to predict profession of participants in E2.
2- apply these models to infer the profession of participants in experiment E1

Why?
Profession categories are strong predictors of the accuracy of tasks. However, E1 does not have this information.
Hence, addingt this to E1 would allow to evaluate how well the causal models that rely on profession generalize 
across the two experiments E1 and E2.

We will compare different models based on different predictors:
- gender
- age
- years of programming
- qualification score
- adjusted qualification score (produced by the Item response models)
