---
layout: post
title: "Image based recommendation(Real time scoring)"
img: malaysia.jpg # Add image post (optional)
date: 2017-09-20 12:53:00 +0300
description: You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. # Add post description (optional)
---
When you have a data that is not labelled, and you do not have feature or variable that you can use directly from columnar database, we will not be able to apply traditional recommendation logics such as association rules or collaborative filtering etc. Here is one such case, where we have images and we want to find similar images in the list. 

The shiny R application displays images and when the user clicks on any image, it generates list of similar images on the bottom of the screen. This happen in real time. 

Internal working: Auto encoders (neural network clustering technique) are used to cluster the features extracted from the images ( 100X100 pixles = 10,000 features). The developed model is kept in H2O. When user clicks on an image the image is scored against model and it sorts the similarly of all other images and selects top 5 to display as a similar image. 
Challenges: The time taken to run the modeling and scoring is about 3 minutes. 
So, to reduce the time, I have separated the modeling process and made it independent. Only scoring happens, now the response time is about 30 seconds. ( Of course still not in the range of  human acceptance but better!). 
Packages used: 
