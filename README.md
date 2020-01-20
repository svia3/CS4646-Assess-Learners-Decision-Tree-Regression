# Class Overview

This was a project completed during the Fall 2019 Semester of my undergraduate career at Georgia Tech, in a class entitled Machine Learning for Trading that taught both basic trading strategies and financial vehicles along with Machine Learning techniques such as Reinforcement Learning, Decision Trees, and Bag Learners. 

# Project Overview 

This project involved cerating a Decision Tree learner, Random Forest, and Bag Learner to learn the trends in the MSCI emerging market index to predict the a buy or sell outcome based on training data. The tree model was built using numpy as follows:

	# TREE 
	# -------------------------------------------------------
	# [ FACTOR | SPLIT_VALUE | LEFT_RELATIVE | RIGHT_RELATIVE ]
	#          |             |               |               
	#          |             |               |               
	#          |             |               |               
	#          |             |               |               

# Implementation

## addEvidence()

  def addEvidence(self, Xtrain, Ytrain):

 This is a wrapper function for recursive function calls used to build the tree.

## buildTree()

  def buildTree(self, data):
  
In order to build our model, training evidence was appended to the the tree and split on the absolute correlation coefficient, calculated using a helper method. This recursive function was used to split the training data on the condition that values in the Xth feature will be to the left of the root caller and greater values will be to the right.  A varying leaf size was used and experimented with to prevent overfitting our training data to the test data. Base cases involve maximum stack depth, the remaining values are all the same with no duplicate, and aggregating the the remaining rows if below the hyper-parameter "leaf size."

## query()

  def query(self, Xpoints):

Using our test data, with Xtest and Ytest broken up appropriately, the tree is traversed according to the split value of a certain node, resulting in a 'Y prediction' that is appended to a list for error checking as compared to 'Ytest.'





