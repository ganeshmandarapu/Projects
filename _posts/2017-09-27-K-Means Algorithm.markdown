---
layout: post
title: "K - Means Algorithm"
img: himalayan.jpg # Add image post (optional)
date: 2017-09-27 12:55:00 +0300
description: You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. # Add post description (optional)
---

#K - means algorithm

K-means is one of the simplest unsupervised learning algorithms that solve the well known clustering problem. 

The procedure follows a simple and easy way to classify a given data set through a certain number of clusters 

K-means algorithm consists of following steps:
1) Place K points into the space represented by the objects that are being clustered. These points represent initial group centroids.

2) Assign each object to the group that has the closest centroid.

3) When all objects have been assigned, recalculate the positions of the K centroids.

4) Repeat Steps 2 and 3 until the centroids no longer move. This produces a separation of the objects into groups from which the metric to be minimized can be calculated.


#R implementation:

Importing the unlabeled data set for performing the algorithm:

```{r}

#distance calculation function 
setwd("S:\\ANALYTICS\\R_Programming\\Clustering")

k_Data <- read.csv("Data_Clustering.csv")

head(k_Data)

dim(k_Data[1,])

```

##Elbow method
```{r}
plot(k_Data, main = "Foods", pch =20, cex =2)
set.seed(7)
km1 = kmeans(k_Data, 2, nstart=100)

# Plot results
plot(k_Data, col =(km1$cluster +1) , main="K-Means result with 2 clusters", pch=20, cex=2)

#Elbow Method

k.max <- 15

data1 <- k_Data

#wss <- (nrow(data1)-1)*sum(apply(data1,2,var))


for (i in 2:15) wss[i] <- sum(kmeans(data1,
                                       centers=i)$withinss)
wss
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares",
     main="Assessing the Optimal Number of Clusters with the Elbow Method",
     pch=20, cex=2)
```



##BIC or AIC

```{r}
library(mclust)

d_clust <- Mclust(as.matrix(k_Data), G=1:20)
m.best <- dim(d_clust$z)[2]
cat("model-based optimal number of clusters:", m.best, "\n")
# 4 clusters
plot(d_clust)
d_clust
```

```{r}
x = k_Data
#Random samples for centers 
centers <- k_Data[sample(nrow(k_Data), 4),]
#centers <- k_Data[18:19,]
```

In the above steps we are considering "x" as our data and "centers" as the randomly initializing centroid. Here in this case we are randomly selecting two data points from the data set itselt as the initial centroid 

Now our next step is to calculate euclidean distances from these initial centroid to the data points. Using the below function we are going to find the distances. We are storing the distances between the points and the initial centroid in distanceMatrix which is 100 * 2 matrix first row consists of values of distances from the first centroid and second rwo consists of values of distance from the second initial centroid this step is scalable even if we initialize more than 2 centroids.  

x is the data matrix with points as rows and dimensions as columns. centers is the matrix of centers (points as rows again). The distanceMatrix is just defines the answer matrix (which will have as many rows as there are rows in the data matrix and as many columns as there re centers). So the point i,j in the result matrix will be the distance from the ith point to the jth center.

Then the for loop iterates over all centers. For each center it computes the euclidean distance from each point to the current center and returns the result.

euclidean distance: sqrt(rowSums(t(t(x)-centers[i,])^2))

```{r}

euclidnew <- function(x, centers) {
  distanceMatrix <- matrix(NA, nrow=nrow(x), ncol=nrow(centers))
  for (j in 1:nrow(centers) ){
    for(i in 1:nrow(x) ) {
      distanceMatrix[i,j] <- sqrt(rowSums((x[i,]-centers[j,])^2))
    }
  }
  distanceMatrix
}

```

Now after calculating the distances from all the points next step is to identify the minimum distances among them and assign with the closet centroid. 

In the below function we are creating two empty vectors for storing cluster and centers history for all the iteration we are going to perform. 

To identify the minimum distances among the distance matrix(row wise) we are using the "apply" fucntion by which we can apply any function to dataset. Here we are using "which.min" function to identify the minimum distances.

Here we perform the last and the fourth step i mentioned above in introduction which we have to repeat the two steps assigning the closest centroids and calculating the centroids of new clusters. This process should be repeated untill the centroid no longer move it's position.

```{r}

K_means <- function(x, centers, distFun, nItter) {
  clusterHistory <- vector(nItter, mode="list")
  centerHistory <- vector(nItter, mode="list")
  
  for(i in 1:nItter) {
    distsToCenters <- distFun(x, centers)
    clusters <- apply(distsToCenters, 1, which.min)
    centers <- apply(x, 2, tapply, clusters, mean)
    
    # Saving history
    clusterHistory[[i]] <- clusters
    centerHistory[[i]] <- centers
  }
  
  list(clusters=clusterHistory, centers=centerHistory)
}

```

In the above function the major step is to assigning the new centroids to the data points which are closed to them. To do this again i used "apply" function to apply the mean to the data points with the new cluster assignment in the previous step. 

The steps followed in this fuction:

1) Finding the euclidean distance with the initial centroids.

2) Identifying the minumum distances and assigning clusters to them.

3) Now repeating the process of finding the centroids for all the data points through all the iterations untill the centroid is no moving. 

Finally we are calling the cluster and center history of number of iteration we are performing as list. 

The clusters are defined by assigning the closest center for each point. And centers are updated as a mean of the points assigned to that center.

Reading the function output into results. Here we can look at the centers and clusters for specific iteration. I presenting the final iteration results below.

```{r} 
Results <- K_means(x, centers, euclidnew, 10)

Results$clusters[[8]]

Results$centers[[8]]
```

Above centers and clusters are the results of the final iteration. 

###Plotting Clusters:
```{r}

library(cluster)
library(fpc)


plotcluster(x, Results$clusters[[8]])

```

###Clulsters Validation

```{r}
library(seriation)

dissplot(dist(x), labels=Results$cluster,
options=list(main="Kmeans Clustering With k=4"))

```




##Python Implementation:

Now implementing the k-means algorithm in python. When compared to R implementation there is no logical difference and the steps performed only syntactical difference which i'm going to explain all the functions and steps i used.

Importing the required libraries and same data set used for R implementation. Here we are using the data frames to implement the algorithm.

```{r engine='python'}
from numpy import *
import numpy as np
import random
from scipy import stats
import pandas as pd
import os

path = "S:\ANALYTICS\Python_learn\Clustering"
os.chdir(path)

k_Data = genfromtxt("Data.csv", delimiter=",")

###k_Data = np.matrix(k_Data)

k_Data = pd.DataFrame(k_Data)

### As in the R implementation we are representing the variables and data set names same. "x" is the data set and "centers" are the initial centroids for which we are using the random.choice function to select the random data points from the data set as centroids. 

x = (k_Data)

centers = k_Data.ix[np.random.choice(k_Data.index, 2)]

###centers = k_Data.iloc[17:19,0:]

### Next step is to determining the euclidean distances between the initial centroids and the data points.

### To slice the data from the initial centroids and and the data set using iloc as it is a data frame. Implementing the same euclidean distance formulae in below function.

def euclidnew(x, centers):
    distanceMatrix = pd.DataFrame(np.nan * np.ones(shape=(x.shape[0],centers.shape[0])))
    for j in range(0, centers.shape[0]):
        for i in range(0, x.shape[0]):
            distanceMatrix.iloc[i][j] = sqrt((centers.iloc[j][0] - x.iloc[i][0])**2 + (centers.iloc[j][1] - x.iloc[i][1])**2)
    #print (distanceMatrix)
    return distanceMatrix
 

### In the final step of python implementation i came up with the "lambda" function(similar to apply function in R) to identify the minimum distances and assigning the clusters to the closest data points.argmin to identify the minimum distances.     

### The additional step i used in below function is adding the newly assigned clusters to the data points through every iteration which will help in the next step to determine the new centers this is because of the lambda functionality.        
### Using group by option with in the lambda function to group the data points on closest cluster point and finding the average to identify new centroids.
    
        
def kmeans(x, centers, euclidnew, niter):
    clusterHistory = [[1]] * 10
    centerHistory =  [[1]] * 10
    
    for i in range (1, niter):
        distsToCenters = euclidnew(x, centers)
        clusters = distsToCenters.apply(lambda x: x.argmin(), axis=1)
        x.loc[:,2] = clusters
        centers = x.groupby(x.loc[:,2]).apply(lambda x:np.average(x.loc[:,0:1], axis=0))
        clusterHistory[i] = clusters
        centerHistory[i] = centers
    return(clusterHistory, centerHistory)
    
kmeans(x, centers, euclidnew, 8)

```



