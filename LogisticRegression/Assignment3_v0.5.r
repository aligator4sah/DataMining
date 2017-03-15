#DM Assignment 2
##Dataset: Audit
##Objective: Predict the binary (TARGET_Adjusted) and continuous (RISK_Adjustment) 
##target variables.

#1--Identify and report the response variables and predictors
#Predictors: Age, Employment, Education, Marital, Occupation, Income, Gender, Deduction, Hours.
#Response variables: TARGET_Adjusted, RISK_Adjustment

#2--Explore data and generate summary
#import the data
audit<-read.csv("C:/Users/daisy/OneDrive/Study/DM/week3/audit.csv", header = TRUE, sep = ",",stringsAsFactors = TRUE)
#Delete the ID column
audit1<-audit[,2:12]
#data summary
dim(audit1)
#There are 11 columns in total. 9 of them are predictors.
summary(audit1)
#The summary shows Age, Income, Deductions, Hours, RISK_Adjustment and TARGET_Adjusted are numerical.
#Employment, Education, Marital, Occupation and Gerder are categotical.
#There are about 100 missing value in Employment and Occupation.
#I would like to impute these missing values with mice packages

#Deal with missing value
library(VIM)
library(mice)
library(ggplot2)
library(gridExtra)
aggr(audit1)

audit2 = audit1

levels(audit2$Employment) = c(levels(audit2$Employment), "NewEmploy")
audit2$Employment[is.na(audit2$Employment)] = "NewEmploy"
summary(audit2$Employment)

levels(audit2$Occupation) = c(levels(audit2$Occupation), "NewOccupy")
audit2$Occupation[is.na(audit2$Occupation)] = "NewOccupy"
summary(audit2$Occupation)

#a--Generate the summary table for data
Age = c(summary(audit2$Age), sd(audit2$Age))
Income = c(summary(audit2$Income), sd(audit2$Income))
Deductions = c(summary(audit2$Deductions), sd(audit2$Deductions))
Hours = c(summary(audit2$Hours), sd(audit2$Hours))
RISK_Adjustment = c(summary(audit2$RISK_Adjustment),sd(audit2$RISK_Adjustment))
result = rbind(Age, Income, Deductions, Hours, RISK_Adjustment)
result = as.data.frame(result)
colnames(result)[7] = c("sd")
library(knitr)
kable(result, caption = 'Table 1: Summary of attributes')

#b--Plot density distribution for numerical variables
#describe whether the variablea have a normal distribution or certain type of skew
library(cowplot)
ggplot(data = audit2, aes(x = Age)) +
  geom_density()

ggplot(data = audit2, aes(x = Income)) +
  geom_density()

ggplot(data = audit2, aes(x = Deductions)) +
  geom_density()

ggplot(data = audit2, aes(x = Hours)) +
  geom_density()

ggplot(data = audit2, aes(x = RISK_Adjustment)) +
  geom_density()

#Except the hours, all the other numeric attributes are skewed to the right.
#Calculate their skewness
library(e1071)
skewness(audit2$Age)

skewness(audit2$Income)

skewness(audit2$Deductions)

skewness(audit2$RISK_Adjustment)

#The skewness of Income, Deductions and RISK_Adjustment are al larger than one,
#which means they are highly skewed to the right, especially Deductions and RISK_Adjustment.
#ONly age's skewness is less than 0.5, which is with tolerance.

#Perform Shapiro-Wilktest, and reject the null hypothesis (normality) if p-value is significant.
shapiro.test(audit2$Age)

shapiro.test(audit2$Income)

shapiro.test(audit2$Deductions)

shapiro.test(audit2$RISK_Adjustment)

#If we set the significance level as 0.05, we can see that all p-values are significant (less than 0.05), 
#which implies that we can reject the null hypothesis and claim that
#all attributes except hours are not normal distribution.

#Draw a normal probability plot (q-q plot), and check if the distribution is approximately forms a straight line.
qqnorm(audit2$Age)
qqline(audit2$Age)

qqnorm(audit2$Income)
qqline(audit2$Income)

qqnorm(audit2$Deductions)
qqline(audit2$Deductions)

qqnorm(audit2$RISK_Adjustment)
qqline(audit2$RISK_Adjustment)
#From q-q plots, we can see that the points for Deductions and RISK_Adjustment
#do not fall on the straight line which clearly voilate the normality assumption. 
#They are skewed to the right.

#c--histogram for categorical variable
ggplot(audit2, aes(x=RISK_Adjustment, fill=Employment))+
  geom_histogram(binwidth = 3000)

ggplot(audit2, aes(x=RISK_Adjustment, fill=Education))+
  geom_histogram(binwidth = 3000)

ggplot(audit2, aes(x=RISK_Adjustment, fill=Marital))+
  geom_histogram(binwidth = 3000)

ggplot(audit2, aes(x=RISK_Adjustment, fill=Occupation))+
  geom_histogram(binwidth = 3000)

ggplot(audit2, aes(x=RISK_Adjustment, fill=Gender))+
  geom_histogram(binwidth = 3000)

#plot_grid(hist_employ, hist_educa, hist_marital, hist_occupa, hist_gender,
#          labels=c("Employment", "Education", "Martial", "Occupation", "Gender"),
#          ncol = 1, nrow = 5)

##3.Apply logistic regression analysis
#a-train logistic regression model
table(audit2$TARGET_Adjusted)

#Implement a 10-fold cross-validation schema
# Cross validation (customized)
library(caret)

Xdel = model.matrix(TARGET_Adjusted~.,data=audit2)[,-1] 
Xdel[1:3,]

n.total=length(audit2$TARGET_Adjusted)
n.train=floor(n.total*(0.9))
n.test=n.total-n.train
train=sample(1:n.total,n.train) ## (randomly) sample indices for training set

xtrain = Xdel[train,]
xtest = Xdel[-train,]
ytrain = audit2$TARGET_Adjusted[train]
ytest = audit2$TARGET_Adjusted[-train]

m1 = glm(TARGET_Adjusted~.,family=binomial,data=data.frame(TARGET_Adjusted=ytrain,xtrain))
summary(m1)


ptest = predict(m1,newdata=data.frame(xtest),type="response")
##  the default predictions are of log-odds (probabilities on logit scale) and type = "response" gives the predicted probabilities
data.frame(ytest,ptest)[1:10,] ## look at the actual value vs. predicted value

btest=floor(ptest+0.5)
confusionMatrix(data=btest, ytest)
conf.matrix = table(ytest,btest)
#Accuracy: 0.96

#Precision
precision=conf.matrix[1,1]/(conf.matrix[1,1]+conf.matrix[1,2])
cat("precision: ",precision)
#Recall
recall=conf.matrix[1,1]/(conf.matrix[1,1]+conf.matrix[2,1])
cat("recall: ",recall)
#F1-score
f1_score=(2*precision*recall)/(precision+recall)
cat("F1-score: ",f1_score)
#AUC
library(pROC)
auc=auc(btest, ptest)
cat("AUC: ",auc)

error=(conf.matrix[1,2]+conf.matrix[2,1])/n.test
error

##Strategy 2
audit3=audit2[c(-3)]
Xdel = model.matrix(TARGET_Adjusted~.,data=audit3)[,-1] 

n.total=length(audit3$TARGET_Adjusted)
n.train=floor(n.total*(0.9))
n.test=n.total-n.train
train=sample(1:n.total,n.train) ## (randomly) sample indices for training set

xtrain = Xdel[train,]
xtest = Xdel[-train,]
ytrain = audit3$TARGET_Adjusted[train]
ytest = audit3$TARGET_Adjusted[-train]

m2 = glm(TARGET_Adjusted~.,family=binomial,data=data.frame(TARGET_Adjusted=ytrain,xtrain))
summary(m2)


ptest = predict(m2,newdata=data.frame(xtest),type="response")
##  the default predictions are of log-odds (probabilities on logit scale) and type = "response" gives the predicted probabilities

btest=floor(ptest+0.5)  ## use floor function to clamp the value to 0 or 1
confusionMatrix(data=btest, ytest)
conf.matrix = table(ytest,btest)
#Accuracy: 0.97

#Precision
precision=conf.matrix[1,1]/(conf.matrix[1,1]+conf.matrix[1,2])
cat("precision: ",precision)
#Recall
recall=conf.matrix[1,1]/(conf.matrix[1,1]+conf.matrix[2,1])
cat("recall: ",recall)
#F1-score
f1_score=(2*precision*recall)/(precision+recall)
cat("F1-score: ",f1_score)
#AUC
library(pROC)
auc=auc(btest, ptest)
cat("AUC: ",auc)

error=(conf.matrix[1,2]+conf.matrix[2,1])/n.test
error

##Strategy 3
audit3=audit2[c(-4)]
Xdel = model.matrix(TARGET_Adjusted~.,data=audit3)[,-1] 

n.total=length(audit3$TARGET_Adjusted)
n.train=floor(n.total*(0.9))
n.test=n.total-n.train
train=sample(1:n.total,n.train) ## (randomly) sample indices for training set

xtrain = Xdel[train,]
xtest = Xdel[-train,]
ytrain = audit3$TARGET_Adjusted[train]
ytest = audit3$TARGET_Adjusted[-train]

m2 = glm(TARGET_Adjusted~.,family=binomial,data=data.frame(TARGET_Adjusted=ytrain,xtrain))
summary(m2)


ptest = predict(m2,newdata=data.frame(xtest),type="response")
##  the default predictions are of log-odds (probabilities on logit scale) and type = "response" gives the predicted probabilities

btest=floor(ptest+0.5)  ## use floor function to clamp the value to 0 or 1
confusionMatrix(data=btest, ytest)
conf.matrix = table(ytest,btest)
#Accuracy: 0.995

#Precision
precision=conf.matrix[1,1]/(conf.matrix[1,1]+conf.matrix[1,2])
cat("precision: ",precision)
#Recall
recall=conf.matrix[1,1]/(conf.matrix[1,1]+conf.matrix[2,1])
cat("recall: ",recall)
#F1-score
f1_score=(2*precision*recall)/(precision+recall)
cat("F1-score: ",f1_score)
#AUC
library(pROC)
auc=auc(btest, ptest)
cat("AUC: ",auc)

error=(conf.matrix[1,2]+conf.matrix[2,1])/n.test
error

#strategy 4
audit3=audit2[c(-8)]
Xdel = model.matrix(TARGET_Adjusted~.,data=audit3)[,-1] 

n.total=length(audit3$TARGET_Adjusted)
n.train=floor(n.total*(0.9))
n.test=n.total-n.train
train=sample(1:n.total,n.train) ## (randomly) sample indices for training set

xtrain = Xdel[train,]
xtest = Xdel[-train,]
ytrain = audit3$TARGET_Adjusted[train]
ytest = audit3$TARGET_Adjusted[-train]

m2 = glm(TARGET_Adjusted~.,family=binomial,data=data.frame(TARGET_Adjusted=ytrain,xtrain))
summary(m2)


ptest = predict(m2,newdata=data.frame(xtest),type="response")
##  the default predictions are of log-odds (probabilities on logit scale) and type = "response" gives the predicted probabilities

btest=floor(ptest+0.5)  ## use floor function to clamp the value to 0 or 1
confusionMatrix(data=btest, ytest)
#0.955


#Naive Bayesian
## determining test and evaluation data sets
n=length(audit2$TARGET_Adjusted)
n1=floor(n*(0.9))
n2=n-n1

train=sample(1:n,n1)

## determining marginal probabilities
#response=as.numeric(levels(audit2$TARGET_Adjusted)[audit2$TARGET_Adjusted])
response = audit2$TARGET_Adjusted

tttt=cbind(audit2$Employment[train],audit2$Education[train],
           audit2$Marital[train],audit2$Occupation[train],
           audit2$Gender[train],response[train])
tttrain0=tttt[tttt[,6]<0.5,]
tttrain1=tttt[tttt[,6]>0.5,]

## prior probabilities
tdel=table(response[train])
tdel=tdel/sum(tdel)
tdel

ts0=table(tttrain0[,1])
ts0=ts0/sum(ts0)
ts0


ts1=table(tttrain1[,1])
ts1=ts1/sum(ts1)
ts1

tc0=table(tttrain0[,2])
tc0=tc0/sum(tc0)
tc0

tc1=table(tttrain1[,2])
tc1=tc1/sum(tc1)
tc1

td0=table(tttrain0[,3])
td0=td0/sum(td0)
td0

td1=table(tttrain1[,3])
td1=td1/sum(td1)
td1

to0=table(tttrain0[,4])
to0=to0/sum(to0)
to0

to1=table(tttrain1[,4])
to1=to1/sum(to1)
to1

tw0=table(tttrain0[,5])
tw0=tw0/sum(tw0)
tw0

tw1=table(tttrain1[,5])
tw1=tw1/sum(tw1)
tw1


## creating test data set
tt=cbind(audit2$Employment[-train],audit2$Education[-train],
         audit2$Marital[-train],audit2$Occupation[-train],
         audit2$Gender[-train],response[-train])

## creating predictions, stored in gg
p0=ts0[tt[,1]]*tc0[tt[,2]]*td0[tt[,3]]*to0[tt[,4]]*tw0[tt[,5]+1]
p1=ts1[tt[,1]]*tc1[tt[,2]]*td1[tt[,3]]*to1[tt[,4]]*tw1[tt[,5]+1]
gg=(p1*tdel[2])/(p1*tdel[2]+p0*tdel[1])
hist(gg)

## coding as 1 if probability 0.5 or larger
gg1=floor(gg+0.5)
ttt=table(response[-train],gg1)
ttt

confusionMatrix(response[-train],gg1)
#Accuracy: 0.8923

error=(ttt[1,2]+ttt[2,1])/n2
error

#Decision Tree
library(MASS) 
library(tree)
library(rpart)

set.seed(1)
train <- sample(1:nrow(audit2), 0.90 * nrow(audit2))

auditTree <- rpart(TARGET_Adjusted ~ . - RISK_Adjustment+Income+Deductions+
                     Hours+Age, data = audit2[train, ], method = 'class')

plot(auditTree)
text(auditTree, pretty = 0)

summary(auditTree)

auditPred <- predict(auditTree, audit2[-train, ], type = 'class')
table(auditPred, audit2[-train, ]$TARGET_Adjusted)
confusionMatrix(auditPred, audit2[-train, ]$TARGET_Adjusted)

#Combine the result of Naive Bayesian and Decision Tree



#For the best model, compute the odds ratio and interpret the effect of each predictors
#Baesd on the accuracy and error, the best model is from logistic model2
summary(m2)

#We can use the confint function to obtain confidence intervals for the coefficient estimates.
confint(m2)

#This is important because the wald.test function refers to the coefficients by their order in the model.
#We use the wald.test function. b supplies the coefficients, 
#while Sigma supplies the variance covariance matrix of the error terms, 
#finally Terms tells R which terms in the model are to be tested.
library(aod)
library(Rcpp)

#odds ratios only
exp(coef(m2))

## odds ratios and 95% CI
exp(cbind(OR = coef(m2), confint(m2)))

#linear and non-linear regression 
#evaluate with leave one out or 10 folds cv
fit1 = lm(RISK_Adjustment ~., data = audit2)
summary(fit1)

audit2$Employment = as.factor(audit2$Employment)
audit2$Occupation = as.factor(audit2$Occupation)

leave.one.out <- function(formula, audit2){
  n = length(audit2$RISK_Adjustment)
  error = dim(n)
  for(k in 1:n){
    id = c(1:n)
    id.train = id[id != k]
    fit = lm(formula, data = audit2[id.train, ])
    predicted = predict(fit, newdata = audit2[-id.train, ])
    observation = audit2$RISK_Adjustment[-id.train]
    error[k] = predicted - observation
  }
  rmse = sqrt(mean(error^2))
  return(rmse)
}


formulaA = RISK_Adjustment ~ .
formulaB = RISK_Adjustment ~ Age+Education+Marital.
formulaC = RISK_Adjustment ~ poly(Income, degree = 2) + poly(Hours, degree = 3) + Age

#Error here: factor Employment has new level Unemployed.
leave.one.out(formulaA, audit2)

