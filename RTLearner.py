

import numpy as np

class RTLearner():

	def __init__(self, leaf_size, verbose=False):
		# Where "leaf_size" is the maximum number of samples to be aggregated at a leaf. 
		# While the tree is being constructed recursively, if there are leaf_size or fewer 
		# elements at the time of the recursive call, the data should be aggregated into a leaf.
		self.leaf_size = leaf_size
		# verbose = True -> can print out information for debugging
		self.verbose = False
		self.tree = None

	def author(self):
		return 'svia3'

	def addEvidence(self, Xtrain, Ytrain):
		# Xtrain and Xtest should be ndarrays (numpy objects) where each row represents 
		# an X1, X2, X3... XN set of feature values. The columns are the features and the rows 
		# are the individual example instances. Y and Ytrain are single dimension ndarrays that 
		# indicate the value we are attempting to predict with X.
		
		#make Ytrain 2-D for concate
		data = np.concatenate((Xtrain, Ytrain[:,None]), axis=1)
		# print(data)
		self.tree = self.buildTree(data)
		# print(self.tree)		

	
	def randomCorrelation(self, data):
		'''
		calculating a random split value
		'''
		return np.random.randint(low=0, high=data.shape[1] - 1)

	def buildTree(self, data):

		if data.shape[0] <= self.leaf_size: #below the limit of leaf_size 
			yMean = np.mean(data[:,-1])
			leafNode = np.array([['leaf', yMean, 'NA', 'NA']])
			return leafNode

		if len(set(data[:,-1])) == 1: # length of the set is 1 -> all the same value with no duplicate
			# only one row left or there are 
			# or all y data is the same
			leafNode = np.array([['leaf', data[0,1], 'NA', 'NA']])
			return leafNode

		i = self.randomCorrelation(data)	 # best feature 
		splitVal = np.median(data[:,i])	 # split the data based off the median of the split feature

		if np.max(data[:,i]) == splitVal:
		# if data[data[:,i] <= splitVal].shape[0] == data.shape[0] or data[data[:,i] > splitVal].shape[0] == data.shape[0]: 
			# stack depth limit / infinite recursion 
			yMean = np.mean(data[:,-1])
			leafNode = np.array([['leaf', yMean, 'NA', 'NA']])
			return leafNode

		leftTree = self.buildTree(data[data[:,i] <= splitVal])
		rightTree = self.buildTree(data[data[:,i] > splitVal]) 		
		root = np.array([i, splitVal, 1, leftTree.shape[0] + 1]) # number of rows of left subtreex
	
		tree = np.vstack((root, leftTree, rightTree))
		return tree

	def query(self, Xpoints):
		yPred = []
		# print(yPred)
		# print(yPred.shape[0])
		for i in range(0, Xpoints.shape[0]): 					# iterate through all data points 
			nodeIndex = 0

			while str(self.tree[nodeIndex,0]) != 'leaf':		# number of rows 
				feature = int(float(self.tree[nodeIndex,0]))		# what is the split feature?
				splitVal = float(self.tree[nodeIndex,1])
				# print(nodeIndex)
				if Xpoints[i, feature] <= splitVal:
					nodeIndex += int(float(self.tree[nodeIndex,2]))     # left relative index 
				else:		# greater than split -> go right
					nodeIndex += int(float(self.tree[nodeIndex,3]))		# right relative index
			# print(type(float(self.tree[nodeIndex,1])))
			yPred.append(float(self.tree[nodeIndex, 1]))   # append the y value 
			# print(nodeIndex) 
		# print(yPred)
		return yPred

	# TREE 
	# -------------------------------------------------------
	# [ FACTOR | SPLIT_VALUE | LEFT_RELATIVE | RIGHT_RELATIVE 
	#          |             |               |               
	#          |             |               |               
	#          |             |               |               
	#          |             |               |               

# if __name__=="__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
#     # print("the secret clue is 'zzyzx'")  
#     print(tree)



