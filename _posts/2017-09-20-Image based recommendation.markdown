---
layout: post
title: "Image based recommendation(Real time scoring)"
img: malaysia.jpg # Add image post (optional)
date: 2017-09-20 12:53:00 +0300
description: You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. # Add post description (optional)
---


Traditional recommendation engine methods such as association rules or collaborative filtering are of no help when you deal with unlabeled data or data without features. Here is one such case, where we have shoe images and we want to find similar shoes that are displayed. Here is image based recommendations prototype.  

The shiny R application displays images and when the user clicks on any image, it generates list of similar shoes on the bottom of the screen. This happens in real time. 

Internal working:
- Auto encoder (neural network clustering technique) is used to cluster the features extracted from the images (100X100 pixels = 10,000 features). The developed model is kept in H2O. When user clicks on a shoe, the shoe image is scored against model and it sorts the similarly of all other shoes and selects top 5 to display as a similar image. 

Challenges:
- The time taken to run the modeling and scoring is about 3 minutes. So, to reduce the time, I have separated the modeling process and made it independent. Only scoring happens in real time, now the response time is about 30 seconds. (Of course, still not in the range of human acceptance but better!). 

Packages used: h20, dplyr, sqldf, caret.

