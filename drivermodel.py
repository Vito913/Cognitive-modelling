### 
### This code is developed by Christian P. Janssen of Utrecht University
### It is intended for students from the Master's course Cognitive Modeling
### Large parts are based on the following research papers:
### Janssen, C. P., & Brumby, D. P. (2010). Strategic adaptation to performance objectives in a dual‐task setting. Cognitive science, 34(8), 1548-1560. https://onlinelibrary.wiley.com/doi/full/10.1111/j.1551-6709.2010.01124.x
### Janssen, C. P., Brumby, D. P., & Garnett, R. (2012). Natural break points: The influence of priorities and cognitive and motor cues on dual-task interleaving. Journal of Cognitive Engineering and Decision Making, 6(1), 5-29. https://journals.sagepub.com/doi/abs/10.1177/1555343411432339
###
### If you want to use this code for anything outside of its intended purposes (training of AI students at Utrecht University), please contact the author:
### c.p.janssen@uu.nl



### 
### import packages
###

import numpy as np
import math

###
###
### Global parameters. These can be called within functions to change (Python: make sure to call GLOBAL)
###
###


###
### Car / driving related parameters
###
steeringUpdateTime = 250    #in ms ## How long does one steering update take? (250 ms consistent with Salvucci 2005 Cognitive Science)
timeStepPerDriftUpdate = 50 ### msec: what is the time interval between two updates of lateral position?
startingPositionInLane = 0.27 			#assume that car starts already slightly away from lane centre (in meters) (cf. Janssen & Brumby, 2010)


#parameters for deviations in car drift due the simulator environment: See Janssen & Brumby (2010) page 1555
gaussDeviateMean = 0
gaussDeviateSD = 0.13 ##in meter/sec


#When the car is actively controlled, calculate a value using equation (1) in Janssen & Brumby (2010). However, some noise is added on top of this equation to account for variation in human behavior. See Janssen & Brumby (2010) page 1555. Also see function "updateSteering" on how this function is used
gaussDriveNoiseMean = 0
gaussDriveNoiseSD = 0.1	#in meter/sec


### The car is controlled using a steering wheel that has a maximum angle. Therefore, there is also a maximum to the lateral velocity coming from a steering update
maxLateralVelocity = 1.7	# in m/s: maximum lateral velocity: what is the maximum that you can steer?
minLateralVelocity = -1* maxLateralVelocity

startvelocity = 0 	#a global parameter used to store the lateral velocity of the car


###
### Switch related parameters
###
retrievalTimeWord = 200   #ms. ## How long does it take to think of the next word when interleaving after a word (time not spent driving, but drifting)
retrievalTimeSentence = 300 #ms. ## how long does it take to retrieve a sentence from memory (time not spent driving, but drifting)



###
### parameters for typing task
###
timePerWord = 0  ### ms ## How much time does one word take
wordsPerMinuteMean = 39.33   # parameters that control typing speed: when typing two fingers, on average you type this many words per minute. From Jiang et al. (2020; CHI)
wordsPerMinuteSD = 10.3 ## this si standard deviation (Jiang et al, 2020)


## Function to reset all parameters. Call this function at the start of each simulated trial. Make sure to reset GLOBAL parameters.
def resetParameters():
    global timePerWord
    global retrievalTimeWord
    global retrievalTimeSentence 
    global steeringUpdateTime 
    global startingPositionInLane 
    global gaussDeviateMean
    global gaussDeviateSD 
    global gaussDriveNoiseMean 
    global gaussDriveNoiseSD 
    global timeStepPerDriftUpdate 
    global maxLateralVelocity 
    global minLateralVelocity 
    global startvelocity
    global wordsPerMinuteMean
    global wordsPerMinuteSD
    
    timePerWord = 0  ### ms

    retrievalTimeWord = 200   #ms
    retrievalTimeSentence = 300 #ms
	
    steeringUpdateTime = 250    #in ms
    startingPositionInLane = 0.27 			#assume that car starts already away from lane centre (in meters)
	

    gaussDeviateMean = 0
    gaussDeviateSD = 0.13 ##in meter/sec
    gaussDriveNoiseMean = 0
    gaussDriveNoiseSD = 0.1	#in meter/sec
    timeStepPerDriftUpdate = 50 ### msec: what is the time interval between two updates of lateral position?
    maxLateralVelocity = 1.7	# in m/s: maximum lateral velocity: what is the maximum that you can steer?
    minLateralVelocity = -1* maxLateralVelocity
    startvelocity = 0 	#a global parameter used to store the lateral velocity of the car
    wordsPerMinuteMean = 39.33
    wordsPerMinuteSD = 10.3

	



##calculates if the car is not accelerating more than it should (maxLateralVelocity) or less than it should (minLateralVelocity)  (done for a vector of numbers)
def velocityCheckForVectors(velocityVectors):
    global maxLateralVelocity
    global minLateralVelocity

    velocityVectorsLoc = velocityVectors

    if (type(velocityVectorsLoc) is list):
            ### this can be done faster with for example numpy functions
        velocityVectorsLoc = velocityVectors
        for i in range(len(velocityVectorsLoc)):
            if(velocityVectorsLoc[i]>1.7):
                velocityVectorsLoc[i] = 1.7
            elif (velocityVectorsLoc[i] < -1.7):
                velocityVectorsLoc[i] = -1.7
    else:
        if(velocityVectorsLoc > 1.7):
            velocityVectorsLoc = 1.7
        elif (velocityVectorsLoc < -1.7):
            velocityVectorsLoc = -1.7

    return velocityVectorsLoc
	




## Function to determine lateral velocity (controlled with steering wheel) based on where car is currently positioned. 
# See Janssen & Brumby (2010) for more detailed explanation. Lateral velocity update depends on current position in lane. Intuition behind function: the further away you are, 
# the stronger the correction will be that a human makes
def vehicleUpdateActiveSteering(LD):

	latVel = 0.2617 * LD*LD + 0.0233 * LD - 0.022
	returnValue = velocityCheckForVectors(latVel)
	return returnValue
	



### function to update steering angle in cases where the driver is NOT steering actively (when they are distracted by typing for example)
def vehicleUpdateNotSteering():
    
    global gaussDeviateMean
    global gaussDeviateSD   

    vals = np.random.normal(loc=gaussDeviateMean, scale=gaussDeviateSD,size=1)[0]
    returnValue = velocityCheckForVectors(vals)
    return returnValue





### Function to run a trial. Needs to be defined by students (section 2 and 3 of assignment)

def runTrial(nrWordsPerSentence =5,nrSentences=3,nrSteeringMovementsWhenSteering=2, interleaving="word"): 
    resetParameters()
    
    locDrift = []
    trialTime = 0
    locDrift.append(startingPositionInLane)
    timePerWord = 60000/wordsPerMinuteMean
    if(interleaving == "word"):
        typingSpeed = np.random.normal(loc=wordsPerMinuteMean, scale=wordsPerMinuteSD,size=100)
        for i in range(nrSentences):
            for k in range(nrWordsPerSentence):
                
                #update steering if its the first word of a first sentance
                if k == 0:
                    typingTime = retrievalTimeWord + timePerWord + retrievalTimeSentence
                    for j in range(math.floor(typingTime/steeringUpdateTime)):
                        vehiclePosition =  locDrift[-1] + vehicleUpdateNotSteering() * steeringUpdateTime * 0.001
                        locDrift.append(vehiclePosition)
                        trialTime = trialTime + steeringUpdateTime ## Typing time probably
                        
                # Update steering when the user is actively steering
                if i != nrSentences-1 or k != nrWordsPerSentence-1:
                    for n in range(nrSteeringMovementsWhenSteering):
                        typingTime = retrievalTimeWord + timePerWord

                        for j in range(math.floor(typingTime/steeringUpdateTime)):
                            drift = vehicleUpdateActiveSteering(locDrift[-1])
                            if(drift >= 0):
                                vehiclePosition = locDrift[-1] - drift * steeringUpdateTime * 0.001
                            else:
                                vehiclePosition = locDrift[-1] + drift * steeringUpdateTime * 0.001
                            locDrift.append(vehiclePosition)
                            trialTime = trialTime + steeringUpdateTime ##
                
                # Update sterering when the user is not actively steering
                else:
                    typingTime = timePerWord + retrievalTimeWord
                    for j in range(math.floor(typingTime/timeStepPerDriftUpdate)):
                        vehiclePosition = locDrift[-1] + vehicleUpdateNotSteering() * (timeStepPerDriftUpdate * 0.001)
                        locDrift.append(vehiclePosition)
                        trialTime = trialTime + steeringUpdateTime ## Typing time probably
            
            trialTime += typingTime
        print(locDrift)
    else:
        print("strategy is not 'word'!")
### function to run multiple simulations. Needs to be defined by students (section 3 of assignment)
def runSimulations(nrSims = 100):
    print("hello world")

runTrial()
