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

library(car)
dt = audit2[,c('Age','Income','Hours','Deductions','RISK_Adjustment')]
cor(dt)

scatterplotMatrix(dt, spread = FALSE, lty.smooth = 2, main = 'Scatter Plot Matrix')

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
library(pROC)
crossvalid <- function(data){
  Xdel = model.matrix(TARGET_Adjusted~.,data=audit3)[,-1] 
  n.total = length(audit3$RISK_Adjustment)
  n.train=floor(n.total*(0.9))
  n.test=n.total-n.train
  error = dim(10)
  accuracy = dim(10)
  precision = dim(10)
  recall = dim(10)
  f1_score = dim(10)
  auc = dim(10)
  for(k in 1:10){
    train=sample(1:n.total,n.train) 
    xtrain = Xdel[train,]
    xtest = Xdel[-train,]
    ytrain = audit3$TARGET_Adjusted[train]
    ytest = audit3$TARGET_Adjusted[-train]
    m1 = glm(TARGET_Adjusted~.,family=binomial,data=data.frame(TARGET_Adjusted=ytrain,xtrain))
    
    ptest = predict(m1,newdata=data.frame(xtest),type="response")
    btest=floor(ptest+0.5)
    conf.matrix = table(ytest,btest)
    accuracy[k] = (conf.matrix[1,1]+conf.matrix[2,2])/n.test
    error[k] = 1-accuracy[k]
    precision[k] = conf.matrix[1,1]/(conf.matrix[1,1]+conf.matrix[1,2])
    recall[k] = conf.matrix[1,1]/(conf.matrix[1,1]+conf.matrix[2,1])
    f1_score[k] = (2*precision*recall)/(precision+recall)
    auc[k] = auc(btest, ptest)
    
  }
  acc_avg = mean(accuracy)
  error_avg = mean(error)
  prec_avg = mean(precision)
  rec_avg = mean(recall)
  f1_avg = mean(f1_score)
  auc_avg = mean(auc)
  cat("accuracy:",acc_avg,"\n")
  cat("error: ",error_avg,"\n")
  cat("precision: ",prec_avg,"\n")
  cat("recall: ",rec_avg,"\n")
  cat("F1_score: ",f1_avg,"\n")
  cat("AUC: ",auc_avg,"\n")
  return(conf.matrix)
}

liftcharts<-function(data){
  Xdel = model.matrix(TARGET_Adjusted~.,data=audit3)[,-1] 
  n.total = length(audit3$RISK_Adjustment)
  n.train=floor(n.total*(0.9))
  n.test=n.total-n.train
  baserate = dim(10)
  for(k in 1:10){
    train=sample(1:n.total,n.train) 
    xtrain = Xdel[train,]
    xtest = Xdel[-train,]
    ytrain = audit3$TARGET_Adjusted[train]
    ytest = audit3$TARGET_Adjusted[-train]
    m1 = glm(TARGET_Adjusted~.,family=binomial,data=data.frame(TARGET_Adjusted=ytrain,xtrain))
    
    ptest = predict(m1,newdata=data.frame(xtest),type="response")
    btest=floor(ptest+0.5)
    df=cbind(ptest,ytest)
    rank.df=as.data.frame(df[order(ptest,decreasing=TRUE),])
    colnames(rank.df) = c('predicted','actual')
    baserate[k]=mean(ytest)
  }
  ax=dim(n.test)
  ay.base=dim(n.test)
  ay.pred=dim(n.test)
  ax[1]=1
  ay.base[1]=mean(baserate)
  ay.pred[1]=rank.df$actual[1]
  for (i in 2:n.test) {
    ax[i]=i
    ay.base[i]=(mean(baserate))*i ## uniformly increase with rate xbar
    ay.pred[i]=ay.pred[i-1]+rank.df$actual[i]
  }
  df=cbind(rank.df,ay.pred,ay.base)
  
  plot(ax,ay.pred,xlab="number of cases",ylab="number of successes",main="Lift: Cum successes sorted by pred val/success prob")
  points(ax,ay.base,type="l")
  return(df)
}

library(ROCR)
ROCcharts<-function(data){
  Xdel = model.matrix(TARGET_Adjusted~.,data=audit3)[,-1] 
  n.total = length(audit3$RISK_Adjustment)
  n.train=floor(n.total*(0.9))
  n.test=n.total-n.train
  sensi = dim(10)
  speci = dim(10)
  for(k in 1:10){
    train=sample(1:n.total,n.train) 
    xtrain = Xdel[train,]
    xtest = Xdel[-train,]
    ytrain = audit3$TARGET_Adjusted[train]
    ytest = audit3$TARGET_Adjusted[-train]
    m1 = glm(TARGET_Adjusted~.,family=binomial,data=data.frame(TARGET_Adjusted=ytrain,xtrain))
    
    ptest = predict(m1,newdata=data.frame(xtest),type="response")
    btest=floor(ptest+0.5)
    cut=1/2
    gg1=floor(ptest+(1-cut))
    truepos =  ytest==1 & ptest>=cut 
    trueneg = ytest==0 & ptest<cut
    sensi[k] = sum(truepos)/sum(ytest==1) 
    speci[k] = sum(trueneg)/sum(ytest==0) 
    data=data.frame(predictions=ptest,labels=ytest)
    pred <- prediction(data$predictions,data$labels)
    perf <- performance(pred, "sens", "fpr")
  }
  plot(perf)
  cat("Specificity:",mean(speci),"\n")
  cat("Sensitivity:",mean(sensi),"\n")
  return(0)
}

audit3 = audit2
crossvalid(audit3)
#0.959
liftcharts(audit3)
ROCcharts(audit3)

##Strategy 2
audit3=audit2[c(-3)]
crossvalid(audit3)
#0.967
liftcharts(audit3)
ROCcharts(audit3)

##Strategy 3
audit3=audit2[c(-4)]
crossvalid(audit3)
#0.963
liftcharts(audit3)
ROCcharts(audit3)

#strategy 4
audit3=audit2[c(-6)]
crossvalid(audit3)
#0.964
liftcharts(audit3)
ROCcharts(audit3)

audit3=audit2[c(-4,-6)]
crossvalid(audit3)
#0.9645
liftcharts(audit3)
ROCcharts(audit3)

audit3=audit2[c(-3,-4,-6)]
crossvalid(audit3)
#0.9575
liftcharts(audit3)
ROCcharts(audit3)

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
dtt = table(auditPred, audit2[-train, ]$TARGET_Adjusted)
confusionMatrix(auditPred, audit2[-train, ]$TARGET_Adjusted)

#Combine the result of Naive Bayesian and Decision Tree



#For the best model, compute the odds ratio and interpret the effect of each predictors
#Baesd on the accuracy and error, the best model is from logistic model2
library(aod)
library(Rcpp)

audit3=audit2[c(-4,-6)]
Xdel = model.matrix(TARGET_Adjusted~.,data=audit3)[,-1] 
xtrain = Xdel
ytrain = audit3$TARGET_Adjusted
m2 = glm(TARGET_Adjusted~.,family=binomial,data=data.frame(TARGET_Adjusted=ytrain,xtrain))

summary(m2)

#We can use the confint function to obtain confidence intervals for the coefficient estimates.
#This is important because the wald.test function refers to the coefficients by their order in the model.
#We use the wald.test function. b supplies the coefficients, 
#while Sigma supplies the variance covariance matrix of the error terms, 
#finally Terms tells R which terms in the model are to be tested.

#odds ratios only
exp(coef(m2))

## odds ratios and 95% CI
exp(cbind(OR = coef(m2), confint(m2)))

varImp(m2)

#linear and non-linear regression 
#evaluate with leave one out or 10 folds cv
audit3=audit2[c(-11)]
fit1 = lm(RISK_Adjustment ~., data = audit3)
summary(fit1)
model.mse = mean(residuals(fit1)^2)
rmse = sqrt(model.mse)
rmse ## r


audit3=audit2[c(-2,-5,-11)]
#I excluded the employment and Occupation two variables because these two factor got 
#some problem. Some of the levels in the factor are too few observations in it
#Thus, they would be taken into new levels in the factor and can't be analyzed.
#Therefore, I excluded these two variables so that we can do leave one out on 
#linear and non-linear regression.

leave.one.out <- function(formula, data){
  n = length(audit3$RISK_Adjustment)
  error = dim(n)
  for(k in 1:n){
    id = c(1:n)
    id.train = id[id != k]
    fit = lm(formula, data = audit3[id.train,])
    predicted = predict(fit, newdata = audit3[-id.train, ])
    observation = audit3$RISK_Adjustment[-id.train]
    error[k] = predicted - observation
  }
  rmse = sqrt(mean(error^2))
  return(rmse)
}


formulaA = RISK_Adjustment ~ Age+Education+Marital+Income+Gender+Hours+Deductions
#7560.038
formulaB = RISK_Adjustment ~ Age+Education+Marital+Hours
#7499.085
formulaC = RISK_Adjustment ~ poly(Hours, degree = 2) + poly(Hours, degree = 3) + Age
#7504.906
formulaD = RISK_Adjustment ~ poly(Hours, degree = 2) + Age
#7503.358
formulaE = RISK_Adjustment ~ poly(Age, degree = 2) + poly(Hours, degree = 4) + Income
#7509.283

leave.one.out(formulaA, audit3)

leave.one.out(formulaB, audit3)

leave.one.out(formulaC, audit3)

leave.one.out(formulaD, audit3)

leave.one.out(formulaE, audit3)

#From the best model, identify the most important predictor in the model, and explain how you determine the importance of the predictors.
leave.one.out(RISK_Adjustment ~ Education+Marital+Hours, data = audit3)
## if remove age
#8109.512

leave.one.out(RISK_Adjustment ~ Age+Marital+Hours, data = audit3) 
## if remove Education
#8148.792

leave.one.out(RISK_Adjustment ~ Age+Education+Hours, data = audit3) 
## if remove Marital
#8200.658

leave.one.out(RISK_Adjustment ~ Age+Education+Marital, data = audit3) 
## if remove Income
#8104.054

leave.one.out(RISK_Adjustment ~ Age+Education+Marital+Income+
                Deductions+Hours+TARGET_Adjusted, data = audit3) 
## if remove Gender
#7559.819

leave.one.out(RISK_Adjustment ~ Age+Education+Marital+Income+
                Gender+Hours+TARGET_Adjusted, data = audit3) 
## if remove Deductions
#7549.482

leave.one.out(RISK_Adjustment ~ Age+Education+Marital+Income+
                Deductions+Gender+TARGET_Adjusted, data = audit3) 
## if remove Hours
#7558.119

leave.one.out(RISK_Adjustment ~ Age+Education+Marital+Income+
                Deductions+Hours+Gender, data = audit3) 
## if remove TARGET_Adjusted
#8106.716

#By removing each one of those predictors and calculating out-of-sample rmse value,
#we can see that removing TARGET_Adjusted will cause the largest rmse
#therefore, TARGET_Adjusted is the most important predictor.
#Actually, by seeing the best model we can also tell that the most important predictor is TARGET_Adjusted
#Since I got the best linear regression model by simply using TARGET_Adjusted predictor.

