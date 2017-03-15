#Assignment 4

#Task: analyze US Senator Roll Call Data
#The objective is to identify and visualize the clustering patterns of senators' voting activities.

#Load the data
library(foreign)
library(ggplot2)

data.url = 'http://www.yurulin.com/class/spring2017_datamining/data/roll_call'
#data.dir = file.path("data", "roll_call")
#data.files = list.files(data.dir)
data.files = c("sen101kh.dta", "sen102kh.dta",
               "sen103kh.dta", "sen104kh.dta",
               "sen105kh.dta", "sen106kh.dta",
               "sen107kh.dta", "sen108kh_7.dta",
               "sen109kh.dta", "sen110kh_2008.dta",
               "sen111kh.dta", "sen112kh.dta",
               "sen113kh.dta")
## Add all roll call vote data frames to a single list
rollcall.data = lapply(data.files,
                       function(f) {
                         read.dta(file.path(data.url, f), convert.factors = FALSE)
                       })
sen113<-read.dta("C:/Users/daisy/OneDrive/Study/DM/week7/sen113kh.dta")

sen113<-as.data.frame(sen113)
#remove the president data
sen113kh = na.omit(sen113)

dim(rollcall.data[[1]])

head(rollcall.data[[1]][,1:12])

## The function takes a single data frame of roll call votes and returns a 
## Senator-by-vote matrix.
rollcall.simplified <- function(df) {
  no.pres <- subset(df, state < 99)
  ## to group all Yea and Nay types together
  for(i in 10:ncol(no.pres)) {
    no.pres[,i] = ifelse(no.pres[,i] > 6, 0, no.pres[,i])
    no.pres[,i] = ifelse(no.pres[,i] > 0 & no.pres[,i] < 4, 1, no.pres[,i])
    no.pres[,i] = ifelse(no.pres[,i] > 1, -1, no.pres[,i])
  }
  
  return(as.matrix(no.pres[,10:ncol(no.pres)]))
}

rollcall.simple = lapply(rollcall.data, rollcall.simplified)

sen113_simple = rollcall.simplified(sen113kh)
## Multiply the matrix by its transpose to get Senator-to-Senator tranformation, 
## and calculate the Euclidan distance between each Senator.
rollcall.dist = lapply(rollcall.simple, function(m) dist(m %*% t(m)))

sen113_dist = dist(sen113_simple %*% t(sen113_simple))

sen113_dist

## Do the MDS
rollcall.mds = lapply(rollcall.dist,
                      function(d) as.data.frame((cmdscale(d, k = 2)) * -1))

## Add identification information about Senators back into MDS data frames
congresses = 101:113

for(i in 1:length(rollcall.mds)) {
  names(rollcall.mds[[i]]) = c("x", "y")
  
  congress = subset(rollcall.data[[i]], state < 99)
  
  congress.names = sapply(as.character(congress$name),
                          function(n) strsplit(n, "[, ]")[[1]][1])
  
  rollcall.mds[[i]] = transform(rollcall.mds[[i]],
                                name = congress.names,
                                party = as.factor(congress$party),
                                congress = congresses[i])
}

head(rollcall.mds[[1]])

## Create a plot of just the 113th Congress
cong.113 <- rollcall.mds[[13]]

base.113 <- ggplot(cong.113, aes(x = x, y = y)) +
  scale_alpha(guide="none") + theme_bw() +
  theme(axis.ticks = element_blank(),
        axis.text.x = element_blank(),
        axis.text.y = element_blank()) +
  xlab("") +
  ylab("") +
  scale_shape(name = "Party", breaks = c("100", "200", "328"),
              labels = c("Dem.", "Rep.", "Ind."), solid = FALSE) +
  scale_color_manual(name = "Party", values = c("100" = "blue",
                                                "200" = "red",
                                                "328"="grey"),
                     breaks = c("100", "200", "328"),
                     labels = c("Dem.", "Rep.", "Ind."))

print(base.113 + geom_point(aes(color = party,
                                alpha = 0.75),size=4))


## Create a single visualization of MDS for all Congresses on a grid
all.mds <- do.call(rbind, rollcall.mds)
all.plot <- ggplot(all.mds, aes(x = x, y = y)) +
  geom_point(aes(color = party, alpha = 0.75), size = 2) +
  scale_alpha(guide="none") +
  theme_bw() +
  theme(axis.ticks = element_blank(),
        axis.text.x = element_blank(),
        axis.text.y = element_blank()) +
  xlab("") +
  ylab("") +
  scale_shape(name = "Party",
              breaks = c("100", "200", "328"),
              labels = c("Dem.", "Rep.", "Ind."),
              solid = FALSE) +
  scale_color_manual(name = "Party", values = c("100" = "blue",
                                                "200" = "red",
                                                "328"="grey"),
                     breaks = c("100", "200", "328"),
                     labels = c("Dem.", "Rep.", "Ind."))+
  facet_wrap(~ congress)

print(all.plot)

####################################################################

## k-means clustering on votes, with k=2 clusters

set.seed(1) ## fix the random seed to produce the same results 
grpSen113_k = kmeans(sen113kh[,c(10:666)], centers=2, nstart=10)
grpSen113_k

library(plyr)
clusterMDS<-function(grpName){
  lbls = sen113kh$name
  party = mapvalues(sen113kh$party,from=c(100, 200, 328),to=c("Dem", "Rep", "Ind") )
  data.mds = cmdscale(sen113_dist)
  
  data2 = data.frame(x=data.mds[,1],y=data.mds[,2],name=lbls,party=party,
                     clu=factor(grpName$cluster))
  
  p = ggplot(aes(x=x,y=y,shape=party,color=clu), data=data2) +
    geom_point(size=6,alpha=0.5) +
    geom_text(aes(x=x,y=y,shape=party,color=clu,label=name), size=4)
  print(p)
  return(data2)
}

kmeans_MDS = clusterMDS(grpSen113_k)


##h_cluster
library(cluster)
hclusterMDS<-function(hc_agg){
  hc1 = cutree(hc_agg,k=2)
  hc1<-as.data.frame(hc1)
  names(hc1)[names(hc1)=="hc1"] <- "cluster"
  data3 = clusterMDS(hc1)
  return(data3)
}

#single link
hc_s = hclust(sen113_dist,method='single')
plot(hc_s)
grpSen2<-hclusterMDS(hc_s)
head(grpSen2)
#complete link
hc_c = hclust(sen113_dist,method='complete')
plot(hc_c)
grpSen3<-hclusterMDS(hc_c)

#average link
hc_a = hclust(sen113_dist,method='average')
plot(hc_a)

grpSen4<-hclusterMDS(hc_a)

######################################################################
#Compare the clustering results with the party labels and
#identify the party members who are assigned to a seemly
#wrong cluster. Specifically, based on the k-means results,
#which Republicans are clustered together with Democrats,
#and vice versa? And based on the three variants
#(single-link, complete-link and average-link), which
#Republicans are clustered together with Democrats, and
#vice versa?

kmeans_MDS = clusterMDS(grpSen113_k)
#Based on the k-means MDS map we can see, cluster 1 should belong to Democrats
#while cluster 2 should belong to Republican.
#That means, if the point is red, then its shape should be a circle. If the point is blue, then its shape should be square.
#However, in this graph, Collins and Chiesa are red square, that means they should be republicans but we clustered them together with Democrats.
#This graph doesn't have any blue circle, that means all Democrats are clustered correctly

grpSen2<-hclusterMDS(hc_s)
#Based on the h-clustering with single link MDS map we can see, cluster 1 should belong to Republican
#while cluster 2 should belong to Dempcrats.
#That means, if the point is red, then its shape should be square. If the point is blue, then its shape should be a circle.
#However, in this graph, Murkowski, Collins and Chiesa are all blue square, that means they should be republicans but we clustered them together with Democrats.
#This graph doesn't have any red circle, that means all Democrats are clustered correctly.

grpSen3<-hclusterMDS(hc_c)
#Based on the h-clustering with complete link MDS map we can see, cluster 1 should belong to Republican
#while cluster 2 should belong to Dempcrats.
#That means, if the point is red, then its shape should be square. If the point is blue, then its shape should be a circle.
#However, in this graph, Murkowski, Collins and Chiesa are all blue square, that means they should be republicans but we clustered them together with Democrats.
#This graph doesn't have any red circle, that means all Democrats are clustered correctly.
#The MDS cluster result is very similar to hclustering with single link.

grpSen4<-hclusterMDS(hc_a)
#Based on the h-clustering with complete link MDS map we can see, cluster 1 should belong to Republican
#while cluster 2 should belong to Dempcrats.
#That means, if the point is red, then its shape should be square. If the point is blue, then its shape should be a circle.
#However, in this graph, Murkowski, Collins and Chiesa are all blue square, that means they should be republicans but we clustered them together with Democrats.
#This graph doesn't have any red circle, that means all Democrats are clustered correctly.
#The three variants of h-clustering show the same result, no Democrats are clusterd wrongly, 
#Murkowski, Collins and Chiesa, this three republican are clustered together with Democrats

###################################################################################
#Compute the purity and entropy for these clustering
#results with respect to the senators' party labels. 

#Evaluation of cluster
head(grpSen2)
head(grpSen3)

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

p1<-cluster.purity(kmeans_MDS$clu,kmeans_MDS$party)
p2<-cluster.purity(grpSen2$clu,grpSen2$party)
p3<-cluster.purity(grpSen3$clu,grpSen3$party)
p4<-cluster.purity(grpSen4$clu,grpSen4$party)

purity<-c(p1,p2,p3,p4)

e1<-cluster.entropy(kmeans_MDS$clu,kmeans_MDS$party)
e2<-cluster.entropy(grpSen2$clu,grpSen2$party)
e3<-cluster.entropy(grpSen3$clu,grpSen3$party)
e4<-cluster.entropy(grpSen4$clu,grpSen4$party)

entropy<-c(e1,e2,e3,e4)

result = rbind(purity,entropy)
result = as.data.frame(result)
colnames(result) = c("k-means", "hclust-single","hclust-complete","hclust-average")

library(knitr)
kable(result, caption = 'Table 1: Summary of Clustering')

##########################################################################
#Based on your observation on both measures and
#mis-classified members, choose two clustering methods
#that generate the most meaningful results and explain why

#Based on the observation and measure table, 
#k-means only have two republican together with democrats and k-means shows the highest purity and lowest entropy
#This means the similarity in each of the k-means cluster is very high and difference inside each cluster is very low.
#Therefore, k-means is the most meaningful results.

#All results from three variants of hierarchical clustering are showing the same. Therefore,I'm going to 
#increase the k value in order to select the best approach in hclust.

hclusterMDS<-function(hc_agg){
  hc1 = cutree(hc_agg,k=3)
  hc1<-as.data.frame(hc1)
  names(hc1)[names(hc1)=="hc1"] <- "cluster"
  data3 = clusterMDS(hc1)
  return(data3)
}

#single link
hc_s = hclust(sen113_dist,method='single')
plot(hc_s)
grpSen2<-hclusterMDS(hc_s)

#complete link
hc_c = hclust(sen113_dist,method='complete')
plot(hc_c)
grpSen3<-hclusterMDS(hc_c)

#average link
hc_a = hclust(sen113_dist,method='average')
plot(hc_a)

grpSen4<-hclusterMDS(hc_a)

p1<-cluster.purity(kmeans_MDS$clu,kmeans_MDS$party)
p2<-cluster.purity(grpSen2$clu,grpSen2$party)
p3<-cluster.purity(grpSen3$clu,grpSen3$party)
p4<-cluster.purity(grpSen4$clu,grpSen4$party)

purity<-c(p1,p2,p3,p4)

e1<-cluster.entropy(kmeans_MDS$clu,kmeans_MDS$party)
e2<-cluster.entropy(grpSen2$clu,grpSen2$party)
e3<-cluster.entropy(grpSen3$clu,grpSen3$party)
e4<-cluster.entropy(grpSen4$clu,grpSen4$party)

entropy<-c(e1,e2,e3,e4)

result = rbind(purity,entropy)
result = as.data.frame(result)
colnames(result) = c("k-means", "hclust-single","hclust-complete","hclust-average")

library(knitr)
kable(result, caption = 'Table 1: Summary of Clustering (k=3)')

#When I increase the k to 3, the hclustering with single method shows the highest purity and
#hclustering with complete method shows the lowest entropy. I would like to choose the hclustering 
#with single moethod to be the second most meaningful clustering method for this dataset.
#since it shows the highest purity and media entropy among the three variants of hclustering method.
