
import numpy as np
import BagLearner as bg 
import LinRegLearner as lrl  	

class InsaneLearner():

	def __init__(self, verbose=False):
		self.insaneLearners = []
		for i in range(0,20):
			self.insaneLearners.append(bg.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20))

	def addEvidence(self, Xtrain, Ytrain):
		for i, learner in enumerate(self.insaneLearners):
			learner.addEvidence(Xtrain=Xtrain,Ytrain=Ytrain)	#adding evidence to bag learners

	def author(self):
		return 'svia3'

	def query(self, Xpoints):
		yPreds = np.empty((Xpoints.shape[0], len(self.insaneLearners))) #always 20
		for k, learner in enumerate(self.insaneLearners): 
			newPred = learner.query(Xpoints)
			yPreds[:,k] = newPred
	
		# print(np.mean(yPreds, axis=1))
		return np.mean(yPreds, axis=1)