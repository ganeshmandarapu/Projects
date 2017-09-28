---
layout: post
title: "Retail Recommendation Engine"
img: indonesia.jpg # Add image post (optional)
date: 2017-09-15 12:51:00 +0300
description: You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. # Add post description (optional)

---



This is a prototype of productionable item based similarity for clothes retailer. Item similarity based recommendations are usually more accurate than user similarity based ones. It is relatively easier to capture item similarity than user similarity due to lot of variety in human preferences. In real world environment ( Consider Amazon recommendations)  similarity indexes are calculated and kept in data frame/ database and when some user express interest (by click) on any item. From the database, the information on similar items is fetched and displayed on the screen. 
This prototype though rudimentary (no real-time scoring). However, it still provides enough complexities of real world scenarios from a coder perspective. This can be directly implemented for any company provided we have got similarity matrix readily available. 
Challenges: 
-	Accomplishing functionalities such as reacting to click. Connecting click as a trigger to database and display.
-	Organizing the tiles and recommendations display in proper order. These can be done in much easier way if we are using Angular Java or any other web development technologies.
Packages used: dplyr, Rshiny, caret, sqldf

