from math import sqrt
import os
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

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
animate_cat = animate.iloc[:, 0:12]
inanimate_cat = inanimate.iloc[:, 0:12]

# Calculate the average value for each voxel in a row

animate_mean = animate_data.mean(axis=1)
inanimate_mean = inanimate_data.mean(axis=1)

# calculate the difference for each value within the animate and inanimate dataframes
overall_mean = pd.DataFrame()
list_a = [] 
for i in range(len(animate_mean)):
    list_a.append(animate_mean.iloc[i] - inanimate_mean.iloc[i])    

overall_mean['mean'] = list_a
    
std_overall = np.std(list_a)
mean_overall = overall_mean.mean()


N = 44
 
t = mean_overall/ (std_overall/ sqrt(N))     
print(t)

# Create a bar plot of average voxel response for animate and inanimate objects.
# Needs to have a error bar added to it to indicate the standard error of the mean

animate_sem = animate_data.sem(axis=1)
inanimate_sem = inanimate_data.sem(axis=1)

plt.bar(1, animate_mean.mean(), yerr=animate_sem.mean(), align='center', alpha=0.5, ecolor='black', capsize=10)
plt.bar(2, inanimate_mean.mean(), yerr=inanimate_sem.mean(), align='center', alpha=0.5, ecolor='black', capsize=10)
plt.xticks([1, 2], ['Animate', 'Inanimate'])
plt.axhline(y=0, color='black')
plt.ylabel('Mean voxel response')
plt.title('Mean voxel response for animate and inanimate objects')
plt.show()

#Use a bar plot to visualize the difference values of the first 20 voxels where the responses should be animate - inanimate
overall_mean = overall_mean.iloc[0:20]
plt.bar(overall_mean.index, overall_mean['mean'], align='center', alpha=0.5, ecolor='black', capsize=10)
plt.axhline(y=0, color='black')
plt.ylabel('Mean voxel response')
plt.title('Mean voxel response for animate and inanimate objects')
plt.show()
