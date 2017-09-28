---
layout: post
title: "Retail Recommendation Engine"
img: sweden.jpg # Add image post (optional)
date: 2017-09-15 12:54:00 +0300
description: You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. # Add post description (optional)
---

In real world recommendation engines work in two different ways 
Item based recommendations are usually more accurate, as it is relatively easier to capture item similarity than user similarity. This shiny app can be considered as a prototype of productionable item based similarity for clothes retailer. 
Usually similarity indexes are calculated and kept in data frame/ database and when some user express interest (by click) on any item. From the database, the information on similar items is fetched and displayed on the screen. 
This prototype though rudimentary (no real time scoring). However, it still provides enough complexities of real world scenarios from a coder perspective. This can be directly implemented for any company provided we have got similarity matrix readily available. 
Challenges: 
-	Accomplishing functionalities such as reacting to click. Connecting click as a trigger to database and display.
-	Organizing the tiles and recommendations display in proper order. These can be done in much easier way if we are using Angular Java or any other web development technologies. 

Packages used: RShiny, Rdashboard
