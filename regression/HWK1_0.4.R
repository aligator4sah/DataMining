#DM Assignment 1
##Dataset: DirectMarketing
##Objective: Explain AmountSpent in terms of the provided customer characteristics

#1--Predictors: Age, Geander, Ownhome, Married, Location, Salary, Children, History, Catalog
#1--Response variables: Amount Spent

#2--Explore data and generate summary
#import the data
DMarket<-read.csv("C:/Study/DM/week2/DirectMarketing.csv", header = TRUE, sep = ",")
#data summary
dim(DMarket)
summary(DMarket)
sum(is.na(DMarket))
#From the summary we can see that, there are 303 missing values in History attributes.
#Since the total record is 1000, 303 missing value is nearly one third of the total amount.
#Therefore, it's better to use the mutiple imputation instead of ignore them.

#a--Deal with missing data

#spell the pattern of missing data
library(VIM)
library(mice)
aggr(DMarket)
#The missing data only appears in History variables and History is a dichotomous variables.
#It is impossible to use avarage or media to impute these missing value.
#Use mice package to impute the missing value

DMarket1<-DMarket

DMarket1$History=as.character(DMarket1$History) #change data type that can add NA
DMarket1[History=='', History:=NA] #set '' in History as missing value
   
summary(DMarket1)
#By using Mutiple imputation, we can keep the History variables to the original distribution.

#b--generate the summary table
sumtable<-table(summary(DMarket1))
sumtable
#name, mean, median, 1st quartile, 3rd quartile, standard deviation.

#c--plot density distribution for Salary and Amount Spent
library(ggplot2) 
library(gridExtra)
library(cowplot)

#ggplot(data = DMarket1, aes(x = Salary)) +
#  geom_histogram(binwidth=3000)

sa_dens<-ggplot(data = DMarket1, aes(x = Salary)) +
  geom_density()

#ggplot(data = DMarket1, aes(x = AmountSpent)) +
#  geom_histogram(binwidth=100) 

Amou_dens<-ggplot(data = DMarket1, aes(x = AmountSpent)) +
  geom_density()

plot_grid(sa_dens, Amou_dens, labels=c("A", "B"), ncol = 2, nrow = 1)

#d--relationship between response variable and numeric predictors
cor(DMarket1$Salary,DMarket1$AmountSpent)
sa_scar<-ggplot(data = DMarket1, aes(x = Salary, y = AmountSpent)) + geom_point()

cor(DMarket1$Children,DMarket1$AmountSpent)
chil_scar<-ggplot(data = DMarket1, aes(x = Children, y = AmountSpent)) + geom_point()

cor(DMarket1$Catalogs,DMarket1$AmountSpent)
cat_scar<-ggplot(data = DMarket1, aes(x = Catalogs, y = AmountSpent)) + geom_point()

plot_grid(sa_scar, chil_scar, cat_scar, labels=c("A", "B", "C"), ncol = 3, nrow = 1)

#e--For each categorical predictor, generate conditional density plot of response variables
#Age and AmountSpent
age_dens<-ggplot(DMarket1, aes(x = AmountSpent, fill = Age, colour = Age)) +
  geom_density(alpha = 0.5) 

#ggplot(DMarket1, aes(x = AmountSpent, fill = Age, colour = Age)) +
#  geom_density(alpha = 0.5) +
#  xlim(200, 20)

#Gender and AmountSpent
gender_dens<-ggplot(DMarket1, aes(x = AmountSpent, fill = Gender, colour = Gender)) +
  geom_density(alpha = 0.5)

#OwnHome and AmountSpent
oh_dens<-ggplot(DMarket1, aes(x = AmountSpent, fill = OwnHome, colour = OwnHome)) +
  geom_density(alpha = 0.5)

#Married and AmountSpent
marr_dens<-ggplot(DMarket1, aes(x = AmountSpent, fill = Married, colour = Married)) +
  geom_density(alpha = 0.5)

#Location and AmountSpent
loca_dens<-ggplot(DMarket1, aes(x = AmountSpent, fill = Location, colour = Location)) +
  geom_density(alpha = 0.5)

#History and AmountSpent
histo_dens<-ggplot(DMarket1, aes(x = AmountSpent, fill = History, colour = History)) +
  geom_density(alpha = 0.5)

plot_grid(age_dens, gender_dens, oh_dens, marr_dens, loca_dens, histo_dens,
          labels=c("A", "B", "C", "D", "E", "F"), ncol = 3, nrow = 2)
#f--For each predictor, compare and describe whether the categories have significantly different means



#3--Apply regression analysis
#a--use all predictors in standard linear regression
Market = as.data.frame(DMarket1)
head(Market)

library(car)
fit = lm(AmountSpent ~ Age+Gender+OwnHome+Married+Location+Salary+Children+History+Catalogs, 
         data=Market)
summary(fit)
#report the model performance with r square, adjust r square and RSME
#Multiple R-squared: 0.7461,  Adjust R-squared: 0.7432
mean.mse = mean((rep(mean(Market$AmountSpent),length(Market$AmountSpent)) - Market$AmountSpent)^2)
model.mse = mean(residuals(fit)^2)
rmse = sqrt(model.mse)
rmse ## root mean square error
#482.5576

#Interprete the regression result


#b--use diffenrent combinations in standard linear and non-linear regression

#standard linear regression
fit1 = lm(AmountSpent ~ Gender+OwnHome+Married+Location+Salary+Children+History+Catalogs, 
          data=Market)
summary(fit1)

fit2 = lm(AmountSpent ~ Age+OwnHome+Married+Location+Salary+Children+History+Catalogs, 
          data=Market)
summary(fit2)

fit3 = lm(AmountSpent ~ Age+Gender+Married+Location+Salary+Children+History+Catalogs, 
          data=Market)
summary(fit3)

#I tried to drop single variables each time, but it turns out, 
#using all the variables shows highest r square value in standard linear regression.
#I think for standard linear regression, using all variables is the best combination.

#Polynomial regression
fit4 = lm(AmountSpent ~ Age+Gender+OwnHome+Married+Location+Salary+Children+History+Catalogs
          + I(Salary^2), data=Market)
summary(fit4)
#The r square turned to 0.7472, which us higher than standard linear regression

fit5 = lm(AmountSpent ~ Age+Gender+OwnHome+Married+Location+Salary+Children+History+Catalogs
          + I(Salary^2)+ I(Children^2) +I(Catalogs^2), data=Market)
summary(fit5)
#The r square turned to 0.7477, which is higher than former one.

#Local Polynomial Regression
library(locfit)
## fit with a 50% nearest neighbor bandwidth.
fitreg=lm(AmountSpent~Salary+Children+Catalogs,data=Market)
plot(AmountSpent~Salary,data=Market)
abline(fitreg)

fit7 <- locfit(AmountSpent~lp(Salary+Children,nn=0.5),data=Market)
fit7
summary(fit7)
plot(fit7)

#Lasso
## the model.matrix statement defines the model to be fitted
x <- model.matrix(AmountSpent~ Age+Gender+OwnHome+Married+Location+Salary+
                    Children+History+Catalogs,data=Market)
x=x[,-1] # stripping off the column of 1s as LASSO includes the intercept automatically
library(lars)
## lasso on all data
lasso <- lars(x=x,y=Market$AmountSpent ,trace=TRUE)
## trace of lasso (standardized) coefficients for varying penalty
plot(lasso)
lasso

#Evaluate which model performs better using out-of-sample RMSE.
#Choose best model using out-of-sample RMSE
## leave-one-out cross validation
n = length(Market$AmountSpent)
error = dim(n)

#standard linear regression model with all response variables
for (k in 1:n) {
  train1 = c(1:n)
  train = train1[train1!=k] ## pick elements that are different from k
  m1 = lm(AmountSpent ~ Age+Gender+OwnHome+Married+Location+Salary+Children+History+Catalogs, data=Market[train ,])
  pred = predict(m1, newdat=Market[-train ,])
  obs = Market$AmountSpent[-train]
  error[k] = obs-pred
}
rmse=sqrt(mean(error^2))
rmse ##489.3011

#fit1 model--delete Age
for (k in 1:n) {
  train1 = c(1:n)
  train = train1[train1!=k] ## pick elements that are different from k
  m2 = lm(AmountSpent ~ Gender+OwnHome+Married+Location+Salary+Children+History+Catalogs, data=Market[train ,])
  pred = predict(m2, newdat=Market[-train ,])
  obs = Market$AmountSpent[-train]
  error[k] = obs-pred
}
rmse=sqrt(mean(error^2))
rmse ##488.6649

#fit2 -- delete Age and OwnHome
for (k in 1:n) {
  train1 = c(1:n)
  train = train1[train1!=k] ## pick elements that are different from k
  m2 = lm(AmountSpent ~ Gender+Married+Location+Salary+Children+History+Catalogs, data=Market[train ,])
  pred = predict(m2, newdat=Market[-train ,])
  obs = Market$AmountSpent[-train]
  error[k] = obs-pred
}
rmse=sqrt(mean(error^2))
rmse ##488.4164

#fit3 -- delete Age, OwnHome, Married
for (k in 1:n) {
  train1 = c(1:n)
  train = train1[train1!=k] ## pick elements that are different from k
  m2 = lm(AmountSpent ~ Gender+Location+Salary+Children+History+Catalogs, data=Market[train ,])
  pred = predict(m2, newdat=Market[-train ,])
  obs = Market$AmountSpent[-train]
  error[k] = obs-pred
}
rmse=sqrt(mean(error^2))
rmse ##488.1256
#This is the smallest RMSE I can get from standard linear regression after trying many combinations.

#fit4 -- polynomial regresssion with all response variables
for (k in 1:n) {
  train1 = c(1:n)
  train = train1[train1!=k] ## pick elements that are different from k
  m2 = lm(AmountSpent ~ Age+Gender+OwnHome+Married+Location+Salary+Children+History+Catalogs
          + I(Salary^2), data=Market[train ,])
  pred = predict(m2, newdat=Market[-train ,])
  obs = Market$AmountSpent[-train]
  error[k] = obs-pred
}
rmse=sqrt(mean(error^2))
rmse ##490.7684

#fit5 -- polynomial regresssion with all some variables and salary square
for (k in 1:n) {
  train1 = c(1:n)
  train = train1[train1!=k] ## pick elements that are different from k
  m2 = lm(AmountSpent ~ Gender+Location+Salary+Children+History+Catalogs
          + I(Salary^2), data=Market[train ,])
  pred = predict(m2, newdat=Market[-train ,])
  obs = Market$AmountSpent[-train]
  error[k] = obs-pred
}
rmse=sqrt(mean(error^2))
rmse ##489.7546

#fit6 -- polynomial regresssion with all some variables and salary square and cube
for (k in 1:n) {
  train1 = c(1:n)
  train = train1[train1!=k] ## pick elements that are different from k
  m2 = lm(AmountSpent ~ Gender+Location+Salary+Children+History+Catalogs
          + I(Salary^2)+I(Salary^3), data=Market[train ,])
  pred = predict(m2, newdat=Market[-train ,])
  obs = Market$AmountSpent[-train]
  error[k] = obs-pred
}
rmse=sqrt(mean(error^2))
rmse ##496.9818
#The best result for polynomial regression is 489.1337 
#after excluding Age, OwnHome and Married variables. And take salary as square and cube.

#fit7 -- local polynomial regression
for (k in 1:n) {
  train1 = c(1:n)
  train = train1[train1!=k] ## pick elements that are different from k
  m2 = locfit(AmountSpent ~ lp(Salary+Children+Catalogs,nn=0.5), data=Market[train ,])
  pred = predict(m2, newdat=Market[-train ,])
  obs = Market$AmountSpent[-train]
  error[k] = obs-pred
}
rmse=sqrt(mean(error^2))
rmse ##695.5365

#c--From the best model, identify the most important predictor
#how to determine the importance of predictor
#consider variable selection in out-of-one sample evaluation setting.

#Use variable selection to test the important predictors to response variable
# backward stepwise selection
library(MASS)
fit1 = lm(AmountSpent ~ Gender+Location+Salary+Children+History+Catalogs,
          data=Market)
stepAIC(fit1, direction="backward")



#The result shows that when using variables including,
#gender, location, salary, children. history and catalogs,
#the AIC will be the smallest


################################3
set.seed(10)
all_data<-data.frame(DMarket1)
positions <- sample(nrow(all_data),size=floor((nrow(all_data)/4)*3))
training<- all_data[positions,]
testing<- all_data[-positions,]

lm_fit<-lm(AmountSpent ~ Gender+Location+Salary+Children+History+Catalogs,data=training)
predictions<-predict(lm_fit,newdata=testing)
error<-sqrt((sum((testing$y-predictions)^2))/nrow(testing))

library(foreach)
length_divisor<-4
iterations<-1000
predictions<-foreach(m=1:iterations,.combine=cbind) %do% {
  training_positions <- sample(nrow(training), size=floor((nrow(training)/length_divisor)))
  train_pos<-1:nrow(training) %in% training_positions
  lm_fit<-lm(AmountSpent ~ Gender+Location+Salary+Children+History+Catalogs,data=training[train_pos,])
  predict(lm_fit,newdata=testing)
}
predictions<-rowMeans(predictions)
error<-sqrt((sum((testing$y-predictions)^2))/nrow(testing))
rmse=sqrt(mean(error^2))
