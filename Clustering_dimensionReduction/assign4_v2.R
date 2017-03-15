#Assignment 4

#Task 1: analyze the data unempstates.csv. The objective of the analysis is to
#group states together if they have similiar trends in unemployment rate.

#Load the data and get the summary
unemp <- read.csv("C:/Users/daisy/OneDrive/Study/DM/week7/unempstates.csv", header = FALSE, sep = ',')
unemp[1:3,]
dim(unemp)

#There are 416 observations in the data set and 50 variables. 50 variables represent 50 states 
#in Unites State and each state is characterized by a feature vector of very large dimension(416),
#with its components representing the 416 monthly observation.

#Get the state names and convert them into column
unemplab<-unemp[1,]
unemplab<-as.data.frame(t(unemplab))



#Impot the data and set the header as True so we can import the data as numeric
unemp1 <- read.csv("C:/Users/daisy/OneDrive/Study/DM/week7/unempstates.csv", header = TRUE, sep = ',')
library(matrixStats)

unemp5<-as.data.frame(t(unemp1))
unemp5<-cbind(unemplab,unemp5)
names(unemp5)[names(unemp5)=="1"] <- "StateName"
#Get the correlation matrix from all variants
cor(unemp5[,-1])
## scale = TRUE: variables are first standardized. Default is FALSE
pcaunemp5 = prcomp(unemp5[,-1], scale=TRUE) 
pcaunemp5

## Get the loadings of PCA
pcaunemp5$rotation

## use 'predict' to project data onto the loadings
unemp5pc = predict(pcaunemp5)
unemp5pc 

#Generate a screeplot and determine the number of principle components
## plot the variances against the number of the principal component;
## retain only factors with eigenvalues greater than 1
plot(pcaunemp5, main="") ## same as screeplot(pcafood)
mtext(side=1, "Unemployment Principal Components",  line=1, font=2)
#The first rectangle's height is almost 225, that means we can choose 225 variables for principle components.


#Plot the loadings for first principal component.
plot(pcaunemp5$rotation[,1],type='l')


#Get the mean value of each state and save them into columns
emean<-colMeans(unemp1)
#Convert the dataframe into matrix and get the standard deviation for each column
unemp_matrix<-as.matrix(unemp1)
esd<-colSds(unemp_matrix)
unemp2<-cbind(unemplab,emean,esd)
unemp2[1:3,]

states = unemp2[,1]
n = length(states)
data = (as.matrix(unemp2[,-1]))
## 1) Center the data according to the mean
my.scaled.data = apply(data,2,function(x) (x-mean(x)))
plot(my.scaled.data,cex=0.9,col="blue",main="Plot of Scaled Data")
## 2) Calculate the covariance matrix
my.cov = cov(my.scaled.data)
my.cov

#3) Calculate the eigenvectors and eigenvalues of the covariance matrix
my.eigen = eigen(my.cov)
my.eigen

#3a) Plot the Eigenvectors over the scaled data
#Generate the scatter plot
plot(my.scaled.data,cex=0.9,col="blue",main="Plot of Scaled Data")
#From the scatter plot we can see, there are two components in the upper side of right corner
#should be excluded. In addtion, there are 2 points on the downside of right corner should be excluded too
#Therefore, there are 47 components should be the right number for the principal components.

#Plot the loadings for the first principal components
pc1.slope = my.eigen$vectors[2,1] /my.eigen$vectors[1,1]
abline(0,pc1.slope,col="red")
#Plot the loadings for the second pricipal components
pc2.slope = my.eigen$vectors[2,2] /my.eigen$vectors[1,2]
abline(0,pc2.slope,col="green")

#Generate a scatterplot to project states on the first two principal components.
## 4) Express the scaled data in terms of eigenvectors (principal components)
## get the P matrix (loadings, i.e., a matrix whose columns contain the eigenvectors)
loadings = my.eigen$vectors 
## project data onto the loadings
scores = my.scaled.data %*% loadings 
## plot the projected data on the first two PCs
plot(scores,ylim=c(-3,3),main='Data in terms of EigenVectors / PCs',xlab='PC1',ylab='PC2')
abline(0,0,col="red")
abline(0,90,col="green")


#MDS map
## calculate distance matrix
MDS<-function(dataset){
unemp_dist = dist(dataset[,-1])
unemp_dist

## visualize clusters
unemp_mds <- cmdscale(unemp_dist)
unemp_mds 

plot(unemp_mds, type = 'n')
text(unemp_mds, labels=dataset[,1])
return(unemp_mds)
}

unemp3<-as.data.frame(t(unemp))
MDS(unemp3)

#clustering preparetion
#Standardzation
#According to the oringal MSD map we can know, there are some outliers in the dataset and therefore, we need to standardize 
#the dataset first to make all the points much tighter clustered.
library(robustHD)
unemp4<-as.data.frame(t(unemp1))
unemp4<-standardize(unemp4) 
unemp4<-cbind(unemplab, unemp4)
names(unemp4)[names(unemp4)=="1"] <- "StateName"
MDS(unemp4)

#As we can see from the MDS map, there are still some points that are far from the central
#We are going to drop this observation so that the rest of the data can be correctly clustered
#Drop WV
#unemp4 = subset(unemp4, !(StateName=="WV") )


#K-mean cluster
#k=4
set.seed(1)  
grpUnemp1 = kmeans(unemp4[,-1], centers=4, nstart=10)
grpUnemp1

## list the cluster assignments
o=order(grpUnemp1$cluster)
data.frame(unemp4$StateName[o],grpUnemp1$cluster[o])

#Adjust the MDS function to make MDS map after clustering
adMDS<-function(dataset,grpName){
  ## calculate distance matrix
  unemp_dist = dist(dataset[,-1])
  unemp_dist
  
  ## visualize clusters
  unemp_mds <- cmdscale(unemp_dist)
  unemp_mds 
  
  plot(unemp_mds, type = 'n')
  text(unemp_mds, labels=dataset[,1],col=grpName$cluster+1)
  return(unemp_mds)
}
#MDS map for k-means with 4 cluster
adMDS(unemp4, grpUnemp1)

#k=8
set.seed(1)  
grpUnemp2 = kmeans(unemp4[,-1], centers=8, nstart=10)
grpUnemp2

## list the cluster assignments
o=order(grpUnemp2$cluster)
data.frame(unemp4$StateName[o],grpUnemp2$cluster[o])

#MDS map for k-means with 8 clusters
adMDS(unemp4, grpUnemp2)

##h-clustering with single-link
library(cluster)

## use hclust,cutree for hierarchical clustering
data.dist = dist(unemp4[,-1]) ## use dist to obtain distance matrix

hc_plot<-function(hc_agg,n){
  hc1 = cutree(hc_agg,k=n)
  hc1<-as.data.frame(hc1)
  names(hc1)[names(hc1)=="hc1"] <- "cluster"
  adMDS(unemp4, hc1)
  return(hc1)
}

#h-clustering with single-link
hc_s = hclust(data.dist,method='single')
plot(hc_s)
#k=4
grpUnemp3<-hc_plot(hc_s,4)
grpUnemp3
#k=8
grpUnemp4<-hc_plot(hc_s,8)
grpUnemp4

#h-clustering with complete-link
hc_c = hclust(data.dist,method='complete')
plot(hc_c)

#k=4
grpUnemp5<-hc_plot(hc_c,4)
grpUnemp5
#k=8
grpUnemp6<-hc_plot(hc_c,8)
grpUnemp6

#h-clustering with average-link
hc_a = hclust(data.dist,method='average')
plot(hc_a)

#k=4
grpUnemp7<-hc_plot(hc_a,4)
grpUnemp7
#k=8
grpUnemp8<-hc_plot(hc_a,8)
grpUnemp8
#two clustering results are most meaningful
#h-clustering with complete-link with 4 cluster
#k-means with 4 cluster

#Evaluation of cluster
cluster.purity <- function(clusters, classes) {
  sum(apply(table(classes, clusters), 2, max)) / length(clusters)
}

cluster.entropy <- function(clusters,classes) {
  en <- function(x) {
    s = sum(x)
    sum(sapply(x/s, function(p) {if (p) -p*log2(p) else 0} ) )
  }
  M = table(classes, clusters)
  m = apply(M, 2, en)
  c = colSums(M) / sum(M)
  sum(m*c)
}


p1<-cluster.purity(grpUnemp1$cluster[o],unemp4$StateName[o])
p2<-cluster.purity(grpUnemp2$cluster[o],unemp4$StateName[o])
p3<-cluster.purity(grpUnemp3$cluster[o],unemp4$StateName[o])
p4<-cluster.purity(grpUnemp4$cluster[o],unemp4$StateName[o])
p5<-cluster.purity(grpUnemp5$cluster[o],unemp4$StateName[o])
p6<-cluster.purity(grpUnemp6$cluster[o],unemp4$StateName[o])
p7<-cluster.purity(grpUnemp7$cluster[o],unemp4$StateName[o])
p8<-cluster.purity(grpUnemp8$cluster[o],unemp4$StateName[o])

purity<-c(p1,p2,p3,p4,p5,p6,p7,p8)

e1<-cluster.entropy(grpUnemp1$cluster[o],unemp4$StateName[o])
e2<-cluster.entropy(grpUnemp2$cluster[o],unemp4$StateName[o])
e3<-cluster.entropy(grpUnemp3$cluster[o],unemp4$StateName[o])
e4<-cluster.entropy(grpUnemp4$cluster[o],unemp4$StateName[o])
e5<-cluster.entropy(grpUnemp5$cluster[o],unemp4$StateName[o])
e6<-cluster.entropy(grpUnemp6$cluster[o],unemp4$StateName[o])
e7<-cluster.entropy(grpUnemp7$cluster[o],unemp4$StateName[o])
e8<-cluster.entropy(grpUnemp8$cluster[o],unemp4$StateName[o])

entropy<-c(e1,e2,e3,e4,e5,e6,e7,e8)

result = rbind(purity,entropy)
result = as.data.frame(result)
colnames(result) = c("k-means4","k-means8", "hc-sig4","hc-sig8","hc-compl4",
                     "hc-compl8","hc-arg4","hc-arg8")

library(knitr)
kable(result, caption = 'Table 1: Summary of Clustering')

library(ggplot2)
library(plyr)
library(reshape2)
result1<-as.data.frame(t(result))
result1$classification<- c("k-means4","k-means8", "hc-sig4","hc-sig8","hc-compl4",
                           "hc-compl8","hc-arg4","hc-arg8")

ggplot(result1, aes(x = classification, y = purity)) + 
  geom_bar(stat = "identity")

ggplot(result1, aes(x = classification, y = entropy)) + 
  geom_bar(stat = "identity")

#hc-complete8, k-means8 are showing good clustering result.
