
import numpy as np
from scipy import stats

class BagLearner():

	def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
		'''
		creating a bag learner object of different learners
		'''
		self.learners = []
		self.bags = bags
		self.boost = boost
		self.verbose = verbose
		self.kwargs = kwargs			#passing in {leaf_size = X}
		for i in range(0, bags):
		    self.learners.append(learner(**kwargs))

	def author(self):
		return 'svia3'

	def addEvidence(self, Xtrain, Ytrain):
		'''
		-adding evidence on a generic learner
		-creating a 3D matrix of random data to parse trees:

		[ [[				[[
			data1 			    data2
					]]				  ]] ]. ... 

		'''
		data = np.concatenate((Xtrain, Ytrain[:,None]), axis=1)

		randData = np.zeros(shape=(data.shape[0],data.shape[1]), dtype=float)			#empty numpy array with colums = data.y
		dataSet = np.zeros(shape=(data.shape[0],data.shape[1], self.bags)) 

		for k in range(dataSet.shape[2]):		#3rd dimension
			for i in range(data.shape[0]):
				randIndex = np.random.randint(low=0, high=Xtrain.shape[0] - 1)
				randData[i,:] = data[randIndex,:]

			dataSet[:,:,k] = randData			# add to the dataSet
		# print(dataSet[:,:,3])

		#build the learner trees
		for k in range(len(self.learners)):
			xData = dataSet[:,0:-1,k] 					#ith randData.x in the dataSet
			yData = np.array(dataSet[:,-1,k]) 			#ith randData.y in the dataSet
			self.learners[k].addEvidence(xData, yData)	#adding evidence and building the tree


	def query(self, Xpoints):

		yPreds = np.empty((Xpoints.shape[0], self.bags))
		# print("Here")
		for k, learner in enumerate(self.learners):
			newPred = learner.query(Xpoints)
			yPreds[:,k] = newPred
	
		# print(np.mean(yPreds, axis=1))
		return stats.mode(yPreds, axis=1)



