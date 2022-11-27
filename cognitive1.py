import matplotlib.pyplot as plt


def start():
    return 0

def perceptual_step(type = "normal"):
    if type == "fast":
        return 50
    elif type == "slow":
        return 200
    elif type == "normal":
        return 100

def cognitive_step(type = "normal"):
    if type == "fast":
        return 25
    elif type == "slow":
        return 170
    elif type == "normal":
        return 70

def motor_step(type = "normal"):
    if type == "fast":
        return 30
    elif type == "slow":
        return 100
    elif type == "normal":
        return 70

def example1():
    total_time = perceptual_step() + cognitive_step() + motor_step()
    print("This took " + str(total_time) + " ms")

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
        
example2("extremes")