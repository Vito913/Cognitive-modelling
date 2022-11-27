import matplotlib.pyplot as plt
import numpy as np

def start():
    return 0

# Time for perceptual step
def perceptual_step(type = "normal"):
    if type == "fast":
        return 50
    elif type == "slow":
        return 200
    elif type == "normal":
        return 100

# Time for cognitive step
def cognitive_step(type = "normal"):
    if type == "fast":
        return 25
    elif type == "slow":
        return 170
    elif type == "normal":
        return 70


# time for motor step
def motor_step(type = "normal"):
    if type == "fast":
        return 30
    elif type == "slow":
        return 100
    elif type == "normal":
        return 70

# first example that returns the time for the whole process
def example1():
    total_time = perceptual_step() + cognitive_step() + motor_step()
    print("This took " + str(total_time) + " ms")


# second task that returns the time for the whole process based on the type of input
def example2(completeness):
    options = ["fast", "slow", "normal"]
    values = []
    if completeness == "extremes":
        fast =   perceptual_step("fast") + cognitive_step("fast") + motor_step("fast")
        slow = perceptual_step("slow") + cognitive_step("slow") + motor_step("slow")
        normal = perceptual_step() + cognitive_step() + motor_step()
        print(fast,normal,slow)
    elif completeness == "all":
        #the function should calculate all possible combinations of processes when each process can be fastman, middleman, or slowman (i.e., 3 steps with each 3 possible values, so 3 x 3 x3 outcomes
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    values.append(start() + perceptual_step(options[i]) + cognitive_step(options[j]) + motor_step(options[k]))
        print(values)
        data1 = values[0 : 8]
        data2 = values[8 : 17]
        data3 = values[18 : 27]
        data = [data1, data2, data3]
        fig = plt.figure(figsize =(10, 7))
        ax = fig.add_axes([0, 0, 1, 1])
        bp = ax.boxplot(data)
        plt.show()
        


# placeholder for the third task

def example3(completeness):
    options = ["fast", "slow", "normal"]
    values = []
    if completeness == "extremes":
        fast = (2* perceptual_step("fast")) + (2* cognitive_step("fast")) + motor_step("fast")
        slow = (2* perceptual_step("slow")) + (2* cognitive_step("slow")) + motor_step("slow")
        normal = (2* perceptual_step()) + (2* cognitive_step()) + motor_step()
        print(fast,normal,slow)
    elif completeness == "all":
        #the function should calculate all possible combinations of processes when each process can be fastman, middleman, or slowman (i.e., 3 steps with each 3 possible values, so 3 x 3 x3 outcomes
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    values.append(start() + perceptual_step(options[i]) + cognitive_step(options[j]) + motor_step(options[k]))
        print(values)

def example4(completeness):
    options = ["fast", "slow", "normal"]
    values = []

    # Your function needs a	for-loop that runs the model for different timings of the second stimulus. 
    # The timing of the second stimulus is relative to time t=0 (trial start) and has options: 40, 80, 110, 150, 210, and 240 ms
    timings = [40, 80, 110, 150, 210, 240]
    for t in timings:
        if completeness == "extremes":
            fast = t + (2* perceptual_step("fast")) + (2* cognitive_step("fast")) + motor_step("fast")
            slow = t + (2* perceptual_step("slow")) + (2* cognitive_step("slow")) + motor_step("slow")
            normal = t + (2* perceptual_step()) + (2* cognitive_step()) + motor_step()
            print(fast,normal,slow)
        elif completeness == "all":
        # the function should calculate all possible combinations of processes when each process can be fastman, middleman, or slowman (i.e., 3 steps with each 3 possible values, so 3 x 3 x3 outcomes
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        values.append(start() + perceptual_step(options[i]) + cognitive_step(options[j]) + motor_step(options[k]) + t)
            print(values)

def example5(completeness):
    # The model starts with a basic error probability of 0.01 (or 1%)
    basic_error = 0.01

    # The total error likelihood is now a multiplication of factors 
    # i. For each middleman step, the error probability is doubled (i.e., x2)
    # ii. For each slowman step, the error probability is halved (i.e.,	x0.5) 
    # iii. For each fastman step, the error probability is tripled (i.e., x3)
    middleman_error_multiplier = 2
    slowman_error_multiplier = 0.5
    fastman_error_multiplier = 3

    options = [("fast", fastman_error_multiplier), ("slow",slowman_error_multiplier), ("normal",middleman_error_multiplier)]
    values = []
    predictions = []
    error_predictions = []

    if completeness == "extremes":
        fast = (2* perceptual_step("fast")) + (2* cognitive_step("fast")) + motor_step("fast")
        slow = (2* perceptual_step("slow")) + (2* cognitive_step("slow")) + motor_step("slow")
        normal = (2* perceptual_step()) + (2* cognitive_step()) + motor_step()
        print(fast,normal,slow)
    
    # Calculate for all possible combinations of fastman, middleman, slowman processes (3 step types x 3 speeds) what the trial time and expected error probability is.
    elif completeness == "all":
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    prediction = start() + perceptual_step(options[i][0]) + cognitive_step(options[j][0]) + motor_step(options[k][0])
                    error_prediction = basic_error * options[i][1] * options[j][1] * options[k][1]
                    predictions.append(prediction)
                    error_predictions.append(error_prediction)
                    values.append(("Time prediction is: " + prediction, "Error prediction is: " + error_prediction))
        print(values)

        N = 50
        x = predictions
        y = error_predictions
        colors = np.random.rand(N)
        area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

        plt.scatter(x, y, s=area, c=colors, alpha=0.5)
        plt.show()