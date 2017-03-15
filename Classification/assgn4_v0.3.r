#Assignment3 
#Objective: predict the binary (TARGET_Adjusted) target variables

#Explore data and generate summary
#import the data
audit<-read.csv("C:/Users/daisy/OneDrive/Study/DM/week3/audit.csv", header = TRUE, sep = ",",stringsAsFactors = TRUE)

#Delete the ID and RISK_Adjustment column
audit1<-audit[,2:12]
audit1<-audit1[,-10]
#data summary
dim(audit1)
#There are 11 columns in total. 9 of them are predictors.
summary(audit1)
#The summary shows Age, Income, Deductions, Hours, RISK_Adjustment and TARGET_Adjusted are numerical.
#Employment, Education, Marital, Occupation and Gerder are categotical.

audit2 <- na.omit(audit1)

summary(audit2)
audit2[1:3,]

#convert TARGET_Ajusted to y so that model can apply to the data
audit3=audit2
audit3$y = as.numeric(audit3$TARGET_Adjusted) 
audit3=audit3[,-10]

#delete the Volunteer and Unemployed levels in Employment and Military level in Occupation
#so that all model can converge
summary(audit3$Employment)
summary(audit3$Occupation)
audit3 = subset(audit3, !(Employment=="Volunteer") )
audit3 = subset(audit3, !(Employment=="Unemployed") )
audit3 = subset(audit3, !(Occupation=="Military"))

levels(droplevels(audit3$Employment))
levels(droplevels(audit3$Occupation))

#########################3
library(MASS) # for the example dataset 
library(plyr) # for recoding data
library(ROCR) # for plotting roc
library(e1071) # for NB and SVM
library(rpart) # for decision tree
library(ada)
library(class)

set.seed(1)

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
           prob = knn(train.set[,-10], test.set[,-10], cl=train.set[,10], k = n, prob=T)
           prob = attr(prob,"prob")
           attr(prob,"prob")[prob==0] = 1-attr(prob,"prob")[prob==0]
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
           test.set = test.set[,-10]
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
                               gamma = 10^(-6:-1), cost = 10^(-1:2))
             summary(tuned)
             gamma = tuned[['best.parameters']]$gamma
             cost = tuned[['best.parameters']]$cost
             model = svm(y~., data = train.set, probability=T, 
                         kernel="radial", gamma=gamma, cost=cost)                        
           }
           test.set = test.set[,-10]
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
#error= 0.3654461 precision= 0.4767403 recall= 0.7855131 
#f-score 0.5099584 auc= 0.8737268 

#knn
audit4 = audit3
audit4$Education = as.numeric(audit3$Education)
audit4$Employment = as.numeric(audit3$Employment)
audit4$Occupation = as.numeric(audit3$Occupation)
audit4$Age = as.numeric(audit3$Age)
audit4$Marital = as.numeric(audit3$Marital)
audit4$Gender = as.numeric(audit3$Gender)
audit4$Hours = as.numeric(audit3$Hours)
#k=2
my.classifier(audit4, cl.name='knn',do.cv=T,n=2)
#error= 0.5394483 precision= 0.2209729 recall= 0.5234899 
#f-score 0.3422148 auc= 0.446959 

#k=3
my.classifier(audit4, cl.name='knn',do.cv=T,n=3)
#error= 0.5714286 precision= 0.2114003 recall= 0.5574944 
#f-score 0.3175966 auc= 0.4328666  

#k=4
my.classifier(audit4, cl.name='knn',do.cv=T,n=4)
#error= 0.5444386 precision= 0.1969807 recall= 0.4657718 
#f-score 0.2861061 auc= 0.4077328 

#k=8
my.classifier(audit4, cl.name='knn',do.cv=T,n=8)
#error= 0.5506721 precision= 0.1840524 recall= 0.4485459 
#f-score 0.2563933 auc= 0.3803317 

#k=2 gets the best result.

#Naive Bayesian
my.classifier(audit3, cl.name='nb',do.cv=T)
#error= 0.3778614 precision= 0.4399146 recall= 0.7759283 
#f-score 0.4949954 auc= 0.8458235

#decision tree
#minimum error
my.classifier(audit3, cl.name='dtree',do.cv=T)
#error= 0.2766034 precision= 0.5466962 recall= 0.6455257 f-score 0.5238818 
#auc= 0.8505014 

#maximum error
do.classification <- function(train.set, test.set, 
                              cl.name, n,verbose=F) {
  switch(cl.name, 
         knn = { # here we test k=3; you should evaluate different k's
           prob = knn(train.set[,-10], test.set[,-10], cl=train.set[,10], k = n, prob=T)
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
           test.set = test.set[,-10]
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
             #pfit<- prune(model, cp=model$cptable[which.min(model$cptable[,"xerror"]),"CP"])
             pfit <-prune(model, cp=model$cptable[which.max(model$cptable[,"xerror"]),"CP"])
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
                               gamma = 10^(-6:-1), cost = 10^(-1:2))
             summary(tuned)
             gamma = tuned[['best.parameters']]$gamma
             cost = tuned[['best.parameters']]$cost
             model = svm(y~., data = train.set, probability=T, 
                         kernel="radial", gamma=gamma, cost=cost)                        
           }
           test.set = test.set[,-10]
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
my.classifier(audit3, cl.name='dtree',do.cv=T)
#error= 0.2864757 precision= 0.5359457 recall= 0.6263982 f-score 0.5002251 
#auc= 0.8405392 

#ada
my.classifier(audit3, cl.name='ada',do.cv=T)
#error= 0.3119899 precision= 0.5165451 recall= 0.7453509 f-score 0.525546 
#auc= 0.8704575 


#SVM
#kernel="radial" gamma = 10^(-6:-1), cost = 10^(-1:2))
my.classifier(audit3, cl.name='svm',do.cv=T)
#error= 0.370126 precision= 0.468805 recall= 0.7755827 
#f-score 0.500315 auc= 0.8607282 

#kernel="linear" gamma = 10^(-6:-1), cost = 10^(1:2))
do.classification <- function(train.set, test.set, 
                              cl.name, n,verbose=F) {
  switch(cl.name, 
         knn = { # here we test k=3; you should evaluate different k's
           prob = knn(train.set[,-10], test.set[,-10], cl=train.set[,10], k = n, prob=T)
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
           test.set = test.set[,-10]
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
                               kernel="linear", 
                               gamma = 10^(-6:-1), cost = 10^(1:2))
             summary(tuned)
             gamma = tuned[['best.parameters']]$gamma
             cost = tuned[['best.parameters']]$cost
             model = svm(y~., data = train.set, probability=T, 
                         kernel="radial", gamma=gamma, cost=cost)                        
           }
           test.set = test.set[,-10]
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
my.classifier(audit3, cl.name='svm',do.cv=T)
#error= 0.3700427 precision= 0.4690666 recall= 0.7757595 
#f-score 0.5003719 auc= 0.8609597 

#kernel="linear" gamma = 10^(-4:-1), cost = 10^(1:2))
do.classification <- function(train.set, test.set, 
                              cl.name, n,verbose=F) {
  switch(cl.name, 
         knn = { # here we test k=3; you should evaluate different k's
           prob = knn(train.set[,-10], test.set[,-10], cl=train.set[,10], k = n, prob=T)
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
           test.set = test.set[,-10]
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
                               kernel="linear", 
                               gamma = 10^(-4:-1), cost = 10^(1:2))
             summary(tuned)
             gamma = tuned[['best.parameters']]$gamma
             cost = tuned[['best.parameters']]$cost
             model = svm(y~., data = train.set, probability=T, 
                         kernel="radial", gamma=gamma, cost=cost)                        
           }
           test.set = test.set[,-10]
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
my.classifier(audit3, cl.name='svm',do.cv=T)
#error= 0.3696428 precision= 0.4706319 recall= 0.7766081 
#f-score 0.5013625 auc= 0.8620705 

#kernel="polynomial" gamma = 10^(-4:-1), cost = 10^(1:2))
do.classification <- function(train.set, test.set, 
                              cl.name, n,verbose=F) {
  switch(cl.name, 
         knn = { # here we test k=3; you should evaluate different k's
           prob = knn(train.set[,-10], test.set[,-10], cl=train.set[,10], k = n, prob=T)
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
           test.set = test.set[,-10]
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
                               kernel="polynomial", 
                               gamma = 10^(-4:-1), cost = 10^(1:2))
             summary(tuned)
             gamma = tuned[['best.parameters']]$gamma
             cost = tuned[['best.parameters']]$cost
             model = svm(y~., data = train.set, probability=T, 
                         kernel="radial", gamma=gamma, cost=cost)                        
           }
           test.set = test.set[,-10]
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
my.classifier(audit3, cl.name='svm',do.cv=T)
#error= 0.3695045 precision= 0.4711515 recall= 0.7769016 
#f-score 0.501859 auc= 0.8624547 

#kernel="sigmoid" gamma = 10^(-4:-1), cost = 10^(1:2))
do.classification <- function(train.set, test.set, 
                              cl.name, n,verbose=F) {
  switch(cl.name, 
         knn = { # here we test k=3; you should evaluate different k's
           prob = knn(train.set[,-10], test.set[,-10], cl=train.set[,10], k = n, prob=T)
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
           test.set = test.set[,-10]
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
                               kernel="sigmoid", 
                               gamma = 10^(-4:-1), cost = 10^(1:2))
             summary(tuned)
             gamma = tuned[['best.parameters']]$gamma
             cost = tuned[['best.parameters']]$cost
             model = svm(y~., data = train.set, probability=T, 
                         kernel="radial", gamma=gamma, cost=cost)                        
           }
           test.set = test.set[,-10]
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
my.classifier(audit3, cl.name='svm',do.cv=T)
#error= 0.3684257 precision= 0.4733574 recall= 0.7791906 
#f-score 0.5037645 auc= 0.8654509

#Make a table of all the variables in the diff models
Accuracy = c(1-0.3654461, 1-0.5394483, 1-0.3778614, 1-0.2766034, 1-0.3119899, 1-0.3684257)
Precision = c(0.4767403,0.2209729, 0.4399146, 0.5466962, 0.5165451, 0.4733574)
Recall = c(0.7855131, 0.5234899, 0.7759283, 0.6455257, 0.7453509, 0.7791906)
F_score = c(0.5099584, 0.3422148, 0.4949954, 0.5238818, 0.525546, 0.5037645)
AUC = c(0.8737268, 0.446959, 0.8458235, 0.8505014, 0.8704575, 0.8654509)
result = rbind(Accuracy,Precision, Recall, F_score, AUC)
result = as.data.frame(result)
colnames(result) = c("logistic_regression","knn","NB","DT","Adaboost","SVM")
library(knitr)
kable(result, caption = 'Table 1: Summary of Classification')
#error= 0.3654461 precision= 0.4767403 recall= 0.7855131 
#f-score 0.5099584 auc= 0.8737268 

#knn:error= 0.5394483 precision= 0.2209729 recall= 0.5234899 
#f-score 0.3422148 auc= 0.446959 

#Nb
#error= 0.3778614 precision= 0.4399146 recall= 0.7759283 
#f-score 0.4949954 auc= 0.8458235

#dt
#error= 0.2766034 precision= 0.5466962 recall= 0.6455257 f-score 0.5238818 
#auc= 0.8505014 

#Ada
#error= 0.3119899 precision= 0.5165451 recall= 0.7453509 f-score 0.525546 
#auc= 0.8704575 

Accuracy1 = c(1-0.3642535,1-0.5350554, 1-0.571534, 1-0.5691443,1-0.3773794, 
              1-0.283924, 1-0.2876411)
Precision1 = c(0.4782371,0.2245224,0.214281,0.2035269, 0.4408058, 0.5365357,0.5298075)
Recall1 = c(0.7880437, 0.5279642, 0.5655481,0.5219985, 0.776951, 0.632392, 0.6562524)
F_score1 = c(0.5114609, 0.3469871, 0.3223208, 0.2998008,0.4958848, 0.5095349, 0.5210202)
AUC1 = c(0.8770393, 0.4602222,0.4396937,0.4164183, 0.8471411, 0.8461313, 0.8444064)
result1 = rbind(Accuracy1,Precision1, Recall1, F_score1, AUC1)
result1= as.data.frame(result1)
colnames(result1) = c("logistic_regression","knn-2","knn-3","knn-4","NB","DT-default",
                      "DT-prune")

Accuracy2 = c(1-0.3019827, 1-0.369064, 1-0.3686135, 1-0.3706021, 1-0.3691601)
Precision2 = c(0.5276435, 0.4729202, 0.4728131, 0.469037, 0.4721092)
Recall2 = c(0.7504386, 0.7778363, 0.7787922, 0.7745726, 0.7776324)
F_score2 = c(0.5354407, 0.5029481, 0.5035869, 0.4998756, 0.5025245)
AUC2 = c(0.8811294, 0.8636782, 0.8649294, 0.859406, 0.8634112)
result2 = rbind(Accuracy2,Precision2, Recall2, F_score2, AUC2)
result2 = as.data.frame(result2)
colnames(result2) = c("Adaboost","SVM-radical","SVM-linear",
                      "SVM-polynomial","SVM-sigmoid")

library(knitr)
kable(result1, caption = 'Table 1: Summary of Classification')

kable(result2, caption = 'Table 2: Summary of Classification')

#F-score bar charts
library(ggplot2)
library(plyr)
library(reshape2)

classification = c("logistic_regression","knn-2","knn-3","knn-4","NB","DT-default",
                   "DT-prune","Adaboost","SVM-radical","SVM-linear", 
                   "SVM-polynomial","SVM-sigmoid")
technique = c("lg","knn","knn","knn","NB","DT","DT","Ada","SVM","SVM","SVM","SVM")
result2 = cbind(F_score, AUC, classification, technique)
result2 = as.data.frame(result2)

ggplot(result2, aes(x = classification, y = F_score, 
                    fill = classification)) + geom_bar(stat = "identity")

#AUC plot
ggplot(result2, aes(x = classification, y = AUC, 
                    fill = technique)) + geom_bar(stat = "identity")

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

legend(x="topright", y=5, legend=c("lr","knn","nb","ada","dt","svm"), lty=c(1,1,1,1), col=c("black","green","red","blue","yellow","purple"),cex=0.75)

