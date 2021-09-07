# ![](https://camo.githubusercontent.com/0eda51f0022b0f98c7050bc3b449b4977d480acf2b8682b086424ccb5361c700/68747470733a2f2f692e696d6775722e636f6d2f5873655855384a2e706e67)

# DSI23 - CAPSTONE PROJECT
Football flight curve prediction
---
# Background and Problem Statement

The is the final project for GA's Data Science Immersive Course.

I am a data scientist currently studying in GA and my friend who is a football (soccer) coach has asked me for help.

He is football coach wants to leverage technology to help promote his business and has asked me to come up with something useful that he can use easily to analyze his client's form and maybe use it on himself.

---
# Executive Summary
Football frames were collected from Youtube and edited using Davinci Resolve for masking purposes. Separate football frames were also taken from Soccer Event dataset (SEV).

Mediapipe API was used as a machine learning solution to extract angle values of human poses in the frames.

OpenCV was used as a computer vision package to enable the running of Mediapipe and also used for object detection and path painting.

Angles were then compiled into a dataframe, and feature engineering was accomplished with PolynomialFeatures, before traditional machine learning models were applied.

This is a classification problem and it is approached by using all classification models found in the Pycaret library.

The top classifier model picked by Pycaret was the extra trees classifier. The following models were then tested for the highest model performance :
1. Extra Trees Classifier
2. Bagged Extra Trees Classfier
3. Boosted Extra Trees Classifier
4. Blended models of Extra Trees, Random Forest and Light Gradient Boost Classifiers

The train, test scores and area under curve of the models were used to gauge the classifier performance.

The criteria for selecting the models are the following:

High Accuracy
Minimize False Negatives (High Recall)
Minimize False Positives (High Precision)
Good Generalization
High recall and high precision is required as we want to get correct predictions.

In summary, Boosted Extra Trees was picked as the chosen classifier with the highest accuracy score of 87%. It had the highest recall score of 0.9067 and the second highest precision score of 0.8679, just shy of the top score of 0.8788. It also had the highest overall score (F1) of 0.8822.

---

## Data Dictionary
#### Dataframes used: combined.csv


|        Variable Name        |    Data Type   |        Description         |
|:---------------------------:|:--------------:|:--------------------------:|
|         rightankle           |      float    |angle of right ankle|
|         leftankle            |      float    |angle of left ankle|
|         rightknee            |      float    |angle of right knee|
|         leftknee             |      float    |angle of left knee|
|         righthip             |      float    |angle of right hip|
|         lefthip              |      float    |angle of left hip|
|         rightshoulder        |      float    |angle of right shoulder|
|         leftshoulder         |      float    |angle of left shoulder|
|         curve                |      int      |label of curve 1 or 0|

---

# Conclusion and Recommendation

There were surprising results using the machine learning classifiers.

As the data scientist involved from the beginning, I was not able to predict with any kind of certainty but boosted extra treees was able to do so.

As the top 4 features included the right ankle as its most important variable. The top five are the following:

1.  rightankle righthip
2.  rightankle rightknee
3.  rightankle leftknee
4.  right leftankle
5.  righthip leftshoulder

All in all the project was fun and was satisfying to accomplish as the machine learning was able to predict with a high amoun of accuracy.

Object tracking and path drawing was also accomplished in this project, providing for good visualk feedback to the user.

However, given the complexity of the project of utilizing computer vision and object tracking, the code for visualization was tailor fit to each video specifically. The code could not be applied to any other video without tweaking some parameters.

There are limitations to the program as well, as the video shown in the visualization was recorded in 120ps, also know as 4x slow motion, mediapipe could work relatively well.

I had test speeding up the video to normal speed and the ball tracking was completely off.

OpenCV was also very specific in video resolution. The resolution of the video will affect speed calculation and also the placement of masks, object detection and tracking, as well as painting.

As such, it is recommended to record videos in slow motion for higher accuracy. Also, the video must be loaded in 1920 by 1080 resolution for now.

Going forward, there are several things that can be done to make the code more user friendly:

1. Allow users to select region of interest for parameter adjustment for video masking could be done so that users can upload other videos they have.
2. Allow users to choose video resolution and then changing parameters to fit

Also, to try and incorporate deep learning for detection purposes. It was not done on this project as tensorflow has some problems running on the Mac M1 chip and video on google colab requires some kind of java scripting.
"# DSI23-Capstone" 
