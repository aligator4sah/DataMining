#Assignment3 
#Objective: predict the binary (TARGET_Adjusted) target variables

#Explore data and generate summary
#import the data
audit<-read.csv("C:/Users/daisy/OneDrive/Study/DM/week3/audit.csv", header = TRUE, sep = ",",stringsAsFactors = TRUE)

#Delete the ID and RISK_Adjustment column
audit1<-audit[,2:12]
audit1<-audit[,-11]
#data summary
dim(audit1)
#There are 11 columns in total. 9 of them are predictors.
summary(audit1)
#The summary shows Age, Income, Deductions, Hours, RISK_Adjustment and TARGET_Adjusted are numerical.
#Employment, Education, Marital, Occupation and Gerder are categotical.

audit2 <- na.omit(audit1)

summary(audit2)
audit2[1:3,]

#audit3=audit2[c(-2,-5)]
audit3=audit2
audit3$y = as.numeric(audit3$TARGET_Adjusted) 
audit3=audit3[,-10]

audit3 = subset(audit3, !(Employment=="Volunteer" || Employment=="Unemployed") )

library(ggplot2)
ggplot(data = audit1, aes(x=Employment)) + 
  geom_bar(stat="count") + 
  scale_x_discrete("Level") + scale_y_continuous("Number") + 
  coord_flip()

audit3$Employment <- as.factor(audit3$Emloyment)
levels(droplevels(audit3$Employment))
levels(audit3$Employment)

#########################3
library(MASS) # for the example dataset 
library(plyr) # for recoding data
library(ROCR) # for plotting roc
library(e1071) # for NB and SVM
library(rpart) # for decision tree
library(ada)
library(class)

set.seed(12345)

my.classifier <- function(dataset, cl.name='knn', do.cv=F, n) {
  n.obs <- nrow(dataset) # no. of observations in dataset
  n.cols <- ncol(dataset) # no. of predictors
  cat('my dataset:',
      n.obs,'observations',
      n.cols-1,'predictors','\n')
  print(dataset[1:3,])
  cat('label (y) distribution:')
  print(table(dataset$y))
  
  pre.test(dataset, cl.name, n)
  if (do.cv) k.fold.cv(dataset, cl.name,n)
}


k.fold.cv <- function(dataset, cl.name, n,k.fold=10, prob.cutoff=0.5) {
  n.obs <- nrow(dataset)
  s = sample(n.obs)
  errors = dim(k.fold)
  probs = NULL
  actuals = NULL
  for (k in 1:k.fold) {
    test.idx = which(s %% k.fold == (k-1) )
    train.set = dataset[-test.idx,]
    test.set = dataset[test.idx,]
    cat(k.fold,'-fold CV run',k,cl.name,':',
        '#training:',nrow(train.set),
        '#testing',nrow(test.set),'\n')
    prob = do.classification(train.set, test.set, cl.name,n)
    predicted = as.numeric(prob > prob.cutoff)
    actual = test.set$y
    confusion.matrix = table(actual,factor(predicted,levels=c(0,1)))
    #confusion.matrix
    error = (confusion.matrix[1,2]+confusion.matrix[2,1]) / nrow(test.set)  
    #errors[k] = error
    cat('\t\terror=',error,'\n')
    probs = c(probs,prob)
    actuals = c(actuals,actual)
    ## you may compute other measures and store them in arrays
  }
  avg.error = mean(errors)
  cat(k.fold,'-fold CV results:','avg error=',avg.error,'\n')
  
  ## plot ROC
  result = data.frame(probs,actuals)
  pred = prediction(result$probs,result$actuals)
  perf = performance(pred, "tpr","fpr")
  plot(perf)  
  
  ## get other measures by using 'performance'
  get.measure <- function(pred, measure.name='auc') {
    perf = performance(pred,measure.name)
    m <- unlist(slot(perf, "y.values"))
    m
  }
  err = mean(get.measure(pred, 'err'))
  precision = mean(get.measure(pred, 'prec'),na.rm=T)
  recall = mean(get.measure(pred, 'rec'),na.rm=T)
  fscore = mean(get.measure(pred, 'f'),na.rm=T)
  cat('error=',err,'precision=',precision,'recall=',recall,'f-score',fscore,'\n')
  auc = get.measure(pred, 'auc')
  cat('auc=',auc,'\n')
}

pre.test <- function(dataset, cl.name, n,r=0.6, prob.cutoff=0.5) {
  n.obs <- nrow(dataset) # no. of observations in dataset
  n.train = floor(n.obs*r)
  train.idx = sample(1:n.obs,n.train)
  train.idx
  train.set = dataset[train.idx,]
  test.set = dataset[-train.idx,]
  cat('pre-test',cl.name,':',
      '#training:',nrow(train.set),
      '#testing',nrow(test.set),'\n')
  colnames(train.set)
  prob = do.classification(train.set, test.set, cl.name,n)
  # prob is an array of probabilities for cases being positive
  length(prob)
  
  ## get confusion matrix
  predicted = as.numeric(prob > prob.cutoff)
  #cat('Predicted:',predicted,'\n')
  actual = test.set$y
  confusion.matrix = table(actual,factor(predicted,levels=c(0,1)))
  error = (confusion.matrix[1,2]+confusion.matrix[2,1]) / nrow(test.set)  
  cat('error rate:',error,'\n')
  # you may compute other measures based on confusion.matrix
  # @see handout03 p.30-
  
  ## plot ROC
  result = data.frame(prob,actual)
  pred = prediction(result$prob,result$actual)
  perf = performance(pred, "tpr","fpr")
  plot(perf)
  return(pred)
}


do.classification <- function(train.set, test.set, 
                              cl.name, n,verbose=F) {
  switch(cl.name, 
         knn = { # here we test k=3; you should evaluate different k's
           prob = knn(train.set[,-9], test.set[,-9], cl=train.set[,9], k = n, prob=T)
           prob = attr(prob,"prob")
           #print(cbind(prob,as.character(test.set$y)))
           prob
         },
         lr = { # logistic regression
           names(train.set)
           
           model = glm(y~., family=binomial, data=train.set)
           if (verbose) {
             print(summary(model))             
           }
           prob = predict(model, newdata=test.set, type="response") 
           #print(cbind(prob,as.character(test.set$y)))
           prob
         },
         nb = {
           
           model = naiveBayes(y~., data=train.set)
           prob = predict(model, newdata=test.set, type="raw") 
           #print(cbind(prob,as.character(test.set$y)))
           prob = prob[,2]/rowSums(prob) # renormalize the prob.
           prob
         },
         dtree = {
           model = rpart(y~., data=train.set)
           test.set = test.set[,-9]
           if (verbose) {
             print(summary(model)) # detailed summary of splits
             printcp(model) # print the cross-validation results
             plotcp(model) # visualize the cross-validation results
             ## plot the tree
             plot(model, uniform=TRUE, main="Classification Tree")
             text(model, use.n=TRUE, all=TRUE, cex=.8)
           }           
           prob = predict(model, newdata=test.set)
           
           if (0) { # here we use the default tree, 
             ## you should evaluate different size of tree
             ## prune the tree 
             pfit<- prune(model, cp=model$cptable[which.min(model$cptable[,"xerror"]),"CP"])
             prob = predict(pfit, newdata=test.set)
             ## plot the pruned tree 
             plot(pfit, uniform=TRUE,main="Pruned Classification Tree")
             text(pfit, use.n=TRUE, all=TRUE, cex=.8)             
           }
           #print(cbind(prob,as.character(test.set$y)))
           head(prob)
           #prob = prob[,2]/rowSums(prob) # renormalize the prob.
           prob
         },
         svm = {
           model = svm(y~., data=train.set, probability=T)
           if (0) { # fine-tune the model with different kernel and parameters
             ## evaluate the range of gamma parameter between 0.000001 and 0.1
             ## and cost parameter from 0.1 until 10
             tuned <- tune.svm(y~., data = train.set, 
                               kernel="radial", 
                               gamma = 10^(-6:-1), cost = 10^(-1:1))
             #print(summary(tuned))
             gamma = tuned[['best.parameters']]$gamma
             cost = tuned[['best.parameters']]$cost
             model = svm(y~., data = train.set, probability=T, 
                         kernel="radial", gamma=gamma, cost=cost)                        
           }
           test.set = test.set[,-9]
           prob = predict(model, newdata=test.set, probability=T)
           dim(prob)
           #prob = attr(prob,"probabilities")
           #print(cbind(prob,as.character(test.set$y)))
           #print(dim(prob))
           
           #prob = prob[,which(colnames(prob)==1)]/colSums(prob)
           #prob = prob[,which(colnames(prob)==1)]/1
           prob
         },
         ada = {
           model = ada(y~., data = train.set)
           prob = predict(model, newdata=test.set, type='probs')
           #print(cbind(prob,as.character(test.set$y)))
           prob = prob[,2]/rowSums(prob)
           prob
         }
  ) 
}


#logistic regression

my.classifier(audit3, cl.name='lr',do.cv=T)
#error= 0.349973 precision= 0.5104301 recall= 0.9153256 
#f-score 0.6007485 auc= 0.9576023

#knn
audit4 = audit3
audit4$Education = as.numeric(audit3$Education)
audit4$Age = as.numeric(audit3$Age)
audit4$Marital = as.numeric(audit3$Marital)
audit4$Gender = as.numeric(audit3$Gender)
audit4$Hours = as.numeric(audit3$Hours)
audit4$RISK_Adjustment = as.numeric(audit3$RISK_Adjustment)
#k=2
my.classifier(audit4, cl.name='knn',do.cv=T,n=2)
#error= 0.5720555 precision= 0.2377846 recall= 0.6480239 
#f-score 0.3820071 auc= 0.5123251 

#k=3
my.classifier(audit4, cl.name='knn',do.cv=T,n=3)
#error= 0.5961032 precision= 0.2376513 recall= 0.6935123 
#f-score 0.3778748 auc= 0.5078361 

#k=5
my.classifier(audit4, cl.name='knn',do.cv=T,n=5)
#error= 0.6421801 precision= 0.233957 recall= 0.756525 
#f-score 0.3715182 auc= 0.5090394

#k=8
my.classifier(audit4, cl.name='knn',do.cv=T,n=8)
#error= 0.6862942 precision= 0.2320476 recall= 0.8218426 
#f-score 0.3682924 auc= 0.5211889 

#k=3 gets the best result.

#Naive Bayesian
my.classifier(audit3, cl.name='nb',do.cv=T)
#error= 0.3400007 precision= 0.534423 recall= 0.8488033 
#f-score 0.5695676 auc= 0.9472671 

#decision tree
my.classifier(audit3, cl.name='dtree',do.cv=T)
#error= 0.3361688 precision= 0.529269 recall= 0.8480468 
#f-score 0.6230341 auc= 0.9315216 

#ada
my.classifier(audit3, cl.name='ada',do.cv=T)
#error= 0.2441541 precision= 0.5770413 recall= 0.9233312 
#f-score 0.6715063 auc= 0.9701908 


#SVM
pre.test <- function(dataset, cl.name, n,r=0.6, prob.cutoff=0.5) {
  n.obs <- nrow(dataset) # no. of observations in dataset
  n.train = floor(n.obs*r)
  train.idx = sample(1:n.obs,n.train)
  train.idx
  train.set = dataset[train.idx,]
  test.set = dataset[-train.idx,]
  cat('pre-test',cl.name,':',
      '#training:',nrow(train.set),
      '#testing',nrow(test.set),'\n')
  colnames(train.set)
  prob = do.classification(train.set, test.set, cl.name,n)
  # prob is an array of probabilities for cases being positive
  length(prob)
  
  ## get confusion matrix
  #predicted = as.numeric(prob > prob.cutoff)
  predicted = as.numeric(prob)
  #cat('Predicted:',predicted,'\n')
  #head(predicted)
  actual = test.set$y
  actual = as.vector(actual)
  predicted = as.vector(predicted)
  confusion.matrix = table(actual,factor(predicted,levels=c(0,1)))
  #confusion.matrix = table(actual,predicted)
  error = (confusion.matrix[1,2]+confusion.matrix[2,1]) / nrow(test.set)  
  cat('error rate:',error,'\n')
  # you may compute other measures based on confusion.matrix
  # @see handout03 p.30-
  
  ## plot ROC
  result = data.frame(prob,actual)
  pred = prediction(result$prob,result$actual)
  perf = performance(pred, "tpr","fpr")
  plot(perf)
  return(pred)
}

k.fold.cv <- function(dataset, cl.name, n,k.fold=10, prob.cutoff=0.5) {
  n.obs <- nrow(dataset)
  s = sample(n.obs)
  errors = dim(k.fold)
  probs = NULL
  actuals = NULL
  for (k in 1:k.fold) {
    test.idx = which(s %% k.fold == (k-1) )
    train.set = dataset[-test.idx,]
    test.set = dataset[test.idx,]
    cat(k.fold,'-fold CV run',k,cl.name,':',
        '#training:',nrow(train.set),
        '#testing',nrow(test.set),'\n')
    prob = do.classification(train.set, test.set, cl.name,n)
    #predicted = as.numeric(prob > prob.cutoff)
    predicted = as.numeric(prob)
    actual = test.set$y
    #confusion.matrix = table(actual,factor(predicted,levels=c(0,1)))
    confusion.matrix = table(actual,predicted)
    #confusion.matrix
    error = (confusion.matrix[1,2]+confusion.matrix[2,1]) / nrow(test.set)  
    #errors[k] = error
    cat('\t\terror=',error,'\n')
    probs = c(probs,prob)
    actuals = c(actuals,actual)
    ## you may compute other measures and store them in arrays
  }
  avg.error = mean(errors)
  cat(k.fold,'-fold CV results:','avg error=',avg.error,'\n')
  
  ## plot ROC
  result = data.frame(probs,actuals)
  pred = prediction(result$probs,result$actuals)
  perf = performance(pred, "tpr","fpr")
  plot(perf)  
  
  ## get other measures by using 'performance'
  get.measure <- function(pred, measure.name='auc') {
    perf = performance(pred,measure.name)
    m <- unlist(slot(perf, "y.values"))
    m
  }
  err = mean(get.measure(pred, 'err'))
  precision = mean(get.measure(pred, 'prec'),na.rm=T)
  recall = mean(get.measure(pred, 'rec'),na.rm=T)
  fscore = mean(get.measure(pred, 'f'),na.rm=T)
  cat('error=',err,'precision=',precision,'recall=',recall,'f-score',fscore,'\n')
  auc = get.measure(pred, 'auc')
  cat('auc=',auc,'\n')
}

my.classifier(audit3, cl.name='svm',do.cv=T)
#error= 0.3334303 precision= 0.5565255 recall= 0.8538208 
#f-score 0.5768728 auc= 0.9629886 

#Make a table of all the variables in the diff models
Accuracy = c(1-0.349973, 1-0.5961032, 1-0.3400007, 1-0.3361688, 1-0.2441541, 1-0.3334303)
Precision = c(0.5104301,0.2376513, 0.534423, 0.529269, 0.5770413, 0.5565255)
Recall = c(0.9153256, 0.6935123, 0.8488033, 0.8480468, 0.9233312, 0.8538208)
F_score = c(0.6007485, 0.3778748, 0.5695676, 0.6230341, 0.6715063, 0.5768728)
AUC = c(0.9576023, 0.5078361, 0.9472671, 0.9315216, 0.9701908, 0.9629886)
result = rbind(Accuracy,Precision, Recall, F_score, AUC)
result = as.data.frame(result)
colnames(result) = c("logistic_regression","knn","NB","DT","Adaboost","SVM")
library(knitr)
kable(result, caption = 'Table 1: Summary of Classification')
#lr: error= 0.349973 precision= 0.5104301 recall= 0.9153256 
#f-score 0.6007485 auc= 0.9576023

#knn:error= 0.5961032 precision= 0.2376513 recall= 0.6935123 
#f-score 0.3778748 auc= 0.5078361 

#Nb
#error= 0.3400007 precision= 0.534423 recall= 0.8488033 
#f-score 0.5695676 auc= 0.9472671 

#ada: error= 0.2441541 precision= 0.5770413 recall= 0.9233312 
#f-score 0.6715063 auc= 0.9701908 

#F-score bar charts
library(ggplot2)
library(plyr)
library(reshape2)

classification = c("logistic_regression","knn","NB","DT","Adaboost","SVM")
result2 = cbind(F_score, AUC, classification)
result2 = as.data.frame(result2)

ggplot(result2, aes(x = classification, y = F_score, 
                    fill = classification)) + geom_bar(stat = "identity")

#AUC plot
ggplot(result2, aes(x = classification, y = AUC, 
                    fill = classification)) + geom_bar(stat = "identity")

#ROC plot
lrr_pred = pre.test(audit3, cl.name = 'lr')
lrr_perf = performance(lrr_pred, "tpr","fpr")

knn_pred = pre.test(audit4, cl.name = "knn", n=3)
knn_perf = performance(knn_pred,"tpr","fpr")

nb_pred = pre.test(audit3, cl.name = 'nb')
nb_perf = performance(nb_pred,"tpr","fpr")

dt_pred = pre.test(audit3, cl.name = 'dtree')
dt_perf = performance(dt_pred,"tpr","fpr")

ada_pred = pre.test(audit3, cl.name = "ada")
ada_perf = performance(ada_pred, "tpr", "fpr")

svm_pred = pre.test(audit3, cl.name = "svm")
svm_perf = performance(svm_pred, "tpr", "fpr")

plot(lrr_perf)

lines(knn_perf@x.values[[1]],knn_perf@y.values[[1]],col='green',lwd=2.5)
lines(nb_perf@x.values[[1]],nb_perf@y.values[[1]],col='red',lwd=2.5)
lines(ada_perf@x.values[[1]],ada_perf@y.values[[1]],col='blue',lwd=2.5)
lines(dt_perf@x.values[[1]],dt_perf@y.values[[1]],col='yellow',lwd=2.5)
lines(lrr_perf@x.values[[1]],lrr_perf@y.values[[1]],col='black',lwd=2.5)
lines(svm_perf@x.values[[1]],svm_perf@y.values[[1]],col='purple',lwd=2.5)

legend(x="topright", y=5, legend=c("lr","knn","nb","ada","dt","svm"), lty=c(1,1,1,1), col=c("black","green","red","blue","yellow","purple"))

