from math import sqrt
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Returns a path that always points to the same file 
def file_path(file_path):
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    abs_file_path = os.path.join(script_dir, file_path)
    return abs_file_path


# Opens the category file and returns an array of the categories "one hot🔥🔥 encoding" model
vectors = file_path("CategoryVectors.txt")
vector_data = open(vectors, 'r')

# Returns the data of the labels for the categories, currently unsued and as the categories were hardcoded
labels = file_path("CategoryLabels.txt")
label_data = open(labels, 'r')

# The categories that the model will be trained on
category_vectors = ["animate","inanimate","human","nonhumani","body","face","natObj","artiObj","rand24","rand48","other48","monkeyape"]


# Get the data from the categoryVectors file and append it to a dataframe
df = pd.DataFrame(data= vector_data)

# separate the data in the dataframe by the "," characters
df = df[0].str.split(",", expand=True)

# Remove the "\n" character from the last column
df[11] = df[11].str.replace("\n", "")

# removes the first row of the dataframe
df = df.iloc[1:]

df.index = df.index -1

# adds the column names to the dataframe
df.columns = category_vectors

# load a different dataframe containing data from NeuralResponses.txt
df2 = pd.read_csv(open(file_path("NeuralResponses_S1.txt"),"r"), sep=",")

# Adds the two dataframes together

df = pd.concat([df, df2], axis=1)


# Creates two dataframes, one for the animate objects and one for the inanimate objects
animate = df.loc[df['animate'] == '1']
inanimate = df.loc[df['inanimate'] == '1']



# split the data where the neural responses are and where the categories are
animate_data = animate.iloc[:, 12:]
inanimate_data = inanimate.iloc[:, 12:]

# Calculate the average value for each voxel in a row

animate_mean = animate_data.mean(axis=1)
inanimate_mean = inanimate_data.mean(axis=1)


 
# calculate the difference for each value within the animate and inanimate dataframes
overall_mean = pd.DataFrame()
list_a = [] 
for i in range(len(animate_mean)):
    list_a.append(animate_mean.iloc[i] - inanimate_mean.iloc[i])    

overall_mean['mean'] = list_a


## Calculate the t-Value with the given formula   
std_overall = np.std(list_a)
mean_overall = overall_mean.mean()


N = 44
 
t = mean_overall/ (std_overall/ sqrt(N))     
print(t)

# Create a bar plot of average voxel response for animate and inanimate objects.
# Needs to have a error bar added to it to indicate the standard error of the mean

# Standard error of the mean is neccessary instead of the standard deveiation

animate_mean = animate_data.mean(axis=1)
inanimate_mean = inanimate_data.mean(axis=1)

animate_sem = animate_mean.sem()
inanimate_sem = inanimate_mean.sem()

plt.bar(1,animate_mean.mean(),yerr= animate_sem.mean(),align="center",alpha=0.7,ecolor="red",capsize=10)
plt.bar(2,inanimate_mean.mean(),yerr=animate_sem.mean(),align="center",alpha=0.7,ecolor="red",capsize=10)
plt.xticks([1,2],["animate","inanimate"])
plt.axhline(y=0,color="black")

plt.ylabel("Response Amplitude")
plt.xlabel("Average Response Amplitude")
plt.show()

#took the mean of all animate of 0 axis and for inanimate then - them get first 20

# plt.bar(first20.index, first20["mean"], align='center', alpha=0.5, ecolor='black', capsize=10)
# plt.axhline(y=0, color='black')
# plt.ylabel('Mean voxel response')
# plt.title('Mean voxel response for animate and inanimate objects')
# plt.show()

# Different axes mean values for the second graph

zero_axis_animate_mean = animate_data.mean(axis=0)
zero_axis_inanimate_mean = inanimate_data.mean(axis=0)
print(zero_axis_animate_mean)
# To get the overall mean we need to - the first from the second mean for each value in the mean

amplitude = np.array(zero_axis_animate_mean - zero_axis_inanimate_mean)
# get the first 20 values from the amplitude array
amplitude = amplitude[0:20]
print(amplitude)

a= []
for i in range(20):
    a.append(i)

print(len(a))
print(len(amplitude))

plt.bar(a,amplitude,align="center",alpha=0.5,ecolor="black",capsize=10)
plt.axhline(y=0,color="black")
plt.xlabel("animate - inanimate")
plt.ylabel("Response amplitude")
plt.show()
