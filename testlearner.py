"""  		   	  			  	 		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		   	  			  	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		   	  			  	 		  		  		    	 		 		   		 		  
All Rights Reserved  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		   	  			  	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		   	  			  	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		   	  			  	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		   	  			  	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		   	  			  	 		  		  		    	 		 		   		 		  
or edited.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		   	  			  	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		   	  			  	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		   	  			  	 		  		  		    	 		 		   		 		  
GT honor code violation.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		   	  			  	 		  		  		    	 		 		   		 		  
"""  		   	  			  	 		  		  		    	 		 		   		 		  
	  			  	 		  		  		    	 		 		   		 		  
import numpy as np  		   	  			  	 		  		  		    	 		 		   		 		  
import math  		   	  			  	 		  		  		    	 		 		   		 		  
import LinRegLearner as lrl  	
import DTLearner as dt
import RTLearner as rt
import BagLearner as bg
import InsaneLearner as it	   	  			  	 		  		  		    	 		 		   		 		  
import sys  
import matplotlib.pyplot as plt 
import time as ti 	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
if __name__=="__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    if len(sys.argv) != 2:  		   	  			  	 		  		  		    	 		 		   		 		  
        print("Usage: python testlearner.py <filename>")  		   	  			  	 		  		  		    	 		 		   		 		  
        sys.exit(1)  		   	  			  	 		  		  		    	 		 		   		 		  
    inf = open(sys.argv[1])  		   	  			  	 		  		  		    	 		 		   		 		  
    data = np.array([list(map(str,s.strip().split(','))) for s in inf.readlines()])  		   	  			  	 		  		  		    	 		 		   		 		  
    #istanbul remove dates and header column
    if(sys.argv[1] == "Data/Istanbul.csv"):
        data = data[1:, 1:]
    data = data.astype('float')
  		   	  			  	 		  		  		    	 		 		   		 		  
    # compute how much of the data is training and testing  		   	  			  	 		  		  		    	 		 		   		 		  
    train_rows = int(0.6* data.shape[0])  		   	  			  	 		  		  		    	 		 		   		 		  
    test_rows = data.shape[0] - train_rows  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # separate out training and testing data  		   	  			  	 		  		  		    	 		 		   		 		  
    trainX = data[:train_rows,0:-1]  		   	  			  	 		  		  		    	 		 		   		 		  
    trainY = data[:train_rows,-1]  		   	  			  	 		  		  		    	 		 		   		 		  
    testX = data[train_rows:,0:-1]  		   	  			  	 		  		  		    	 		 		   		 		  
    testY = data[train_rows:,-1]  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"TrainX shape: {trainX.shape}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"TrainY shape: {trainY.shape}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"TextX shape: {testX.shape}")                                                                
    print(f"TextY shape: {testY.shape}")                                                                
  		   	  			  	 		  		  		    	 		 		   		 		  
    # create a learner and train it  		   	  			  	 		  		  		    	 		 		   		 		  
    # learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner  		   	  			  	 		  		  		    	 		 		   		 		  
    # learner.addEvidence(trainX, trainY) # train it  		   	  			  	 		  		  		    	 		 		   		 		  
    # print(learner.author())  		   	  			  	 		  		  		    	 		 		   		 		  
  
    #-----------------------------------------------------------------------------------------------
    # Create a DTlearner and train it
    # -----------------------------------------------------------------------------------------------
    learner = dt.DTLearner(leaf_size=1, verbose=False) # constructor
    learner.addEvidence(trainX, trainY) # training step
    # predY = learner.query(testX) # query		 
    print("--------DT LEARNER--------") 	

    # evaluate in sample  (USING THE TRAINING DATA) 	   	  			  	 		  		  		    	 		 		   		 		  
    predY = learner.query(trainX) # get the predictions  		   	  			  	 		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])  		   	  			  	 		  		  		    	 		 		   		 		     	  			  	 		  		  		    	 		 		   		 		  
    print("In sample results")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		   	  			  	 		  		  		    	 		 		   		 		  
    c = np.corrcoef(predY, y=trainY)  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # evaluate out of sample  	 (USING THE TEST DATA)	   	  			  	 		  		  		    	 		 		   		 		  
    predY = learner.query(testX) # get the predictions  		   	  			  	 		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print("Out of sample results")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		   	  			  	 		  		  		    	 		 		   		 		  
    c = np.corrcoef(predY, y=testY)  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")  	
    # -----------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------
    # Create a RTlearner and train it
    # -----------------------------------------------------------------------------------------------
    learner = rt.RTLearner(leaf_size=1, verbose=False) # constructor
    learner.addEvidence(trainX, trainY) # training step
    # predY = learner.query(testX) # query     
    print("--------RT LEARNER--------")   

    # evaluate in sample  (USING THE TRAINING DATA)                                                               
    predY = learner.query(trainX) # get the predictions                                                               
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])                                                                                                                         
    print("In sample results")                                                                
    print(f"RMSE: {rmse}")                                                                
    c = np.corrcoef(predY, y=trainY)                                                                
    print(f"corr: {c[0,1]}")                                                                
                                                                
    # evaluate out of sample     (USING THE TEST DATA)                                                            
    predY = learner.query(testX) # get the predictions                                                                
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])                                                               
    print()                                                               
    print("Out of sample results")                                                                
    print(f"RMSE: {rmse}")                                                                
    c = np.corrcoef(predY, y=testY)                                                               
    print(f"corr: {c[0,1]}")    
    # -----------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------
    # create a Bag Learner and train it
    # -----------------------------------------------------------------------------------------------
    learner = bg.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size":1}, bags=20, boost=False, verbose=False)
    learner.addEvidence(trainX, trainY)
    # Y = learner.query(testX)
    print("--------BAG LEARNER--------") 

     # evaluate in sample  (USING THE TRAINING DATA)       
    predY = learner.query(trainX) # get the predictions                                                               
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])                                                                                                                         
    print("In sample results")                                                                
    print(f"RMSE: {rmse}")                                                                
    c = np.corrcoef(predY, y=trainY)                                                                
    print(f"corr: {c[0,1]}")                                                                
                                                                
    # evaluate out of sample     (USING THE TEST DATA)                                                            
    predY = learner.query(testX) # get the predictions                                                                
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])                                                               
    print()                                                               
    print("Out of sample results")                                                                
    print(f"RMSE: {rmse}")                                                                
    c = np.corrcoef(predY, y=testY)                                                               
    print(f"corr: {c[0,1]}")    
    # -----------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------
    # create an Insane Learner and train it
    # -----------------------------------------------------------------------------------------------
    learner = it.InsaneLearner(verbose=False) # constructor
    learner.addEvidence(trainX, trainY) # training step

    # Y = learner.query(testX)
    print("--------INSANE LEARNER--------") 

     # evaluate in sample  (USING THE TRAINING DATA)       
    predY = learner.query(trainX) # get the predictions                                                               
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])                                                                                                                         
    print("In sample results")                                                                
    print(f"RMSE: {rmse}")                                                                
    c = np.corrcoef(predY, y=trainY)                                                                
    print(f"corr: {c[0,1]}")                                                                
                                                                
    # evaluate out of sample     (USING THE TEST DATA)                                                            
    predY = learner.query(testX) # get the predictions                                                                
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])                                                               
    print()                                                               
    print("Out of sample results")                                                                
    print(f"RMSE: {rmse}")                                                                
    c = np.corrcoef(predY, y=testY)                                                               
    print(f"corr: {c[0,1]}")    

    # -----------------------------------------------------------------------------------------------
    # report question number 1
    # -----------------------------------------------------------------------------------------------
    print("--------GRAPHS FOR QUESTION 1--------")
    inSampleRSME = [0] # add dummy value so that leaf size of 1 is at index 1
    outSampleRSME = [0] # add dummy value so that leaf size of 1 is at index 1
    for i in range(1,51): #0-50 leaf sizes
        # create the learner with varying leaf size
        learner = dt.DTLearner(leaf_size=i, verbose=False) 
        learner.addEvidence(trainX, trainY) # training step
        # IN SAMPLE-------------------------------------------
        predY = learner.query(trainX) # get the predictions                                                               
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0]) 
        inSampleRSME.append(rmse)    #build in-sample RSME data                                                               
        # OUT SAMPLE-------------------------------------------                                                          
        predY = learner.query(testX) # get the predictions                                                                
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])                                                                                                                         
        outSampleRSME.append(rmse)

    inSampleRSME
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.plot(inSampleRSME)
    ax.plot(outSampleRSME)
    ax.set_title('Overfitting assessed by RSME vs. LeafSize using DTLearners')
    # ax.legend(('In_Sample","Out_Sample'))
    ax.set_ylabel('RSME')
    ax.set_xlabel('Leaf_Size')
    ax.set_xlim(1, 50) # set the xlim
    # dim = np.arange(1,50,5); # get your locations
    # ax.set_xticks(dim)  # set the locations of the xticks to be on the integers
    # ax.grid()    # turn the grid on
    ax.legend(('In_Sample', 'Out_Sample'), loc='lower right')
    plt.savefig('report_question1.png')
    # plt.show()

     # -----------------------------------------------------------------------------------------------
    # report question number 1
    # -----------------------------------------------------------------------------------------------
    print("--------GRAPHS FOR QUESTION 2--------")
    inSampleRSME = [0] # add dummy value so that leaf size of 1 is at index 1
    outSampleRSME = [0] # add dummy value so that leaf size of 1 is at index 1
    for i in range(1,51): #1-50 leaf sizes
        # create the learner with varying leaf size
        learner = bg.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size":i}, bags=20, boost=False, verbose=False)
        learner.addEvidence(trainX, trainY)
        # IN SAMPLE-------------------------------------------
        predY = learner.query(trainX) # get the predictions                                                               
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0]) 
    
        inSampleRSME.append(rmse)    #build in-sample RSME data                                                               
        # OUT SAMPLE-------------------------------------------                                                          
        predY = learner.query(testX) # get the predictions                                                                
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])                                                                                                                         
        outSampleRSME.append(rmse)

    inSampleRSME
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.plot(inSampleRSME)
    ax.plot(outSampleRSME)
    ax.set_title('Overfitting assessed by RSME vs. LeafSize using a BagLearner with bags=20 of DTLearners')
    # ax.legend(('In_Sample","Out_Sample'))
    ax.set_ylabel('RSME')
    ax.set_xlabel('Leaf_Size')
    ax.set_xlim(1, 50) # set the xlim
    # dim = np.arange(1,50,5); # get your locations
    # ax.set_xticks(dim)  # set the locations of the xticks to be on the integers
    # ax.grid()    # turn the grid on
    ax.legend(('In_Sample', 'Out_Sample'), loc='lower right')
    plt.savefig('report_question2.png')
    # plt.show()

    # -----------------------------------------------------------------------------------------------
    # report question number 3
    # -----------------------------------------------------------------------------------------------
    print("--------GRAPHS FOR QUESTION 3--------")
    DT_In_Sample_MPE = [0] # add dummy value so that leaf size of 1 is at index 1
    # DT_Out_Sample_MAPE = [0] # add dummy value so that leaf size of 1 is at index 1
    RT_In_Sample_MPE = [0] # add dummy value so that leaf size of 1 is at index 1
    # RT_Out_Sample_MAPE = [0] # add dummy value so that leaf size of 1 is at index 1
    for i in range(1,51): #1-50 leaf sizes
        # create the learner with varying leaf size
        learner = dt.DTLearner(leaf_size=i, verbose=False) 
        learner.addEvidence(trainX, trainY) # training step
        # IN SAMPLE-------------------------------------------
        predY = learner.query(trainX) # get the predictions  
        MPE = np.mean((trainY - predY) / trainY) * 100 / len(trainY)                                                         
        DT_In_Sample_MPE.append(MPE)                                     
        # OUT SAMPLE------------------------------------------- 
        # predY = learner.query(testX) # get the predictions                                                       
        # MAPE = np.mean(np.abs(((testY - predY) / testY)) * (100 / len(testY)))                                                         
        # DT_Out_Sample_MAPE.append(MAPE)                                                                  
                                                              
        learner = rt.RTLearner(leaf_size=i, verbose=False) 
        learner.addEvidence(trainX, trainY) # training step                                                                                                              
        # IN SAMPLE-------------------------------------------
        predY = learner.query(trainX) # get the predictions  
        MPE = np.mean((trainY - predY) / trainY) * 100 / len(trainY)                                                       
        RT_In_Sample_MPE.append(MPE)                                     
        # OUT SAMPLE------------------------------------------- 
        # predY = learner.query(testX) # get the predictions                                                       
        # MAPE = np.mean(np.abs(((testY - predY) / testY)) * (100 / len(testY)))                                                         
        # RT_Out_Sample_MAPE.append(MAPE)   

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.plot(DT_In_Sample_MPE, color='r')
    # ax.plot(DT_Out_Sample_MAPE)
    ax.plot(RT_In_Sample_MPE, color='g')
    # ax.plot(RT_Out_Sample_MAPE)
    ax.set_title('MPE vs. Leaf Size for DTLearners vs. RTLearners Using Training Data')
    # ax.legend(('In_Sample","Out_Sample'))
    ax.set_ylabel('MPE')
    ax.set_xlabel('Leaf_Size')
    ax.set_xlim(1, 50) # set the xlim
    # dim = np.arange(1,50,5); # get your locations
    # ax.set_xticks(dim)  # set the locations of the xticks to be on the integers
    # ax.grid()    # turn the grid on
    ax.legend(('DT_In_Sample_MPE', 'RT_In_Sample_MPE'), loc='lower right')
    plt.savefig('report_question3_metric1.png')
    # plt.show()
    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # second metric
    # -----------------------------------------------------------------------------------------------
    DT_In_Sample_TIME = [0] # add dummy value so that leaf size of 1 is at index 1
    # DT_Out_Sample_MAPE = [0] # add dummy value so that leaf size of 1 is at index 1
    RT_In_Sample_TIME = [0] # add dummy value so that leaf size of 1 is at index 1
    # RT_Out_Sample_MAPE = [0] # add dummy value so that leaf size of 1 is at index 1
    for i in range(1,51): #1-50 leaf sizes
        # create the learner with varying leaf size
        # IN SAMPLE------------------------------------------- 
        learner = dt.DTLearner(leaf_size=i, verbose=False) 
        start = ti.time()
        learner.addEvidence(trainX, trainY) # training step                                     
        DT_In_Sample_TIME.append(ti.time() - start)

        learner = rt.RTLearner(leaf_size=i, verbose=False) 
        start = ti.time()
        learner.addEvidence(trainX, trainY) # training step                                     
        RT_In_Sample_TIME.append(ti.time() - start) 
      
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.plot(DT_In_Sample_TIME, color='m')
    # ax.plot(DT_Out_Sample_MAPE)
    ax.plot(RT_In_Sample_TIME, color='y')
    # ax.plot(RT_Out_Sample_MAPE)
    ax.set_title('Time vs. Leaf Size for DTLearners vs. RTLearners Using Training Data')
    # ax.legend(('In_Sample","Out_Sample'))
    ax.set_ylabel('Time (seconds)')
    ax.set_xlabel('Leaf_Size')
    ax.set_xlim(1, 50) # set the xlim
    # dim = np.arange(1,50,5); # get your locations
    # ax.set_xticks(dim)  # set the locations of the xticks to be on the integers
    # ax.grid()    # turn the grid on
    ax.legend(('DT_In_Sample_TIME', 'RT_In_Sample_TIME'), loc='upper right')
    plt.savefig('report_question3_metric2.png')
    # plt.show()




