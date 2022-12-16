from math import sqrt
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

# Returns a path that always points to the same file 
def file_path(file_path):
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    abs_file_path = os.path.join(script_dir, file_path)
    return abs_file_path


#============================Initiate variables==========================================#

# Opens the category file and returns an array of the categories "one hot🔥🔥 encoding" model
vectors = file_path("CategoryVectors.txt")
vector_data = open(vectors, 'r')

# Returns the data of the labels for the categories, currently unsued and as the categories were hardcoded
labels = file_path("CategoryLabels.txt")
label_data = open(labels, 'r')

# The categories that the model will be trained on
category_vectors = np.array(["animate","inanimate","human","nonhumani","body","face","natObj","artiObj","rand24","rand48","other48","monkeyape"])

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

df_full = pd.concat([df, df2], axis=1)


# Creates two dataframes, one for the animate objects and one for the inanimate objects
animate = df_full.loc[df['animate'] == '1']
inanimate = df_full.loc[df['inanimate'] == '1']



# split the data where the neural responses are and where the categories are
animate_data = animate.iloc[:, 12:]
inanimate_data = inanimate.iloc[:, 12:]

# Calculate the average value for each voxel in a row

animate_mean = animate_data.mean(axis=1)
inanimate_mean = inanimate_data.mean(axis=1)

zero_axis_animate_mean = animate_data.mean(axis=0)
zero_axis_inanimate_mean = inanimate_data.mean(axis=0)
# To get the overall mean we need to - the first from the second mean for each value in the mean

amplitude = np.array(zero_axis_animate_mean - zero_axis_inanimate_mean)
# get the first 20 values from the amplitude array
amplitude = amplitude[0:20]

a= []
for i in range(20):
    a.append(i)

a = np.array(a)

 
# calculate the difference for each value within the animate and inanimate dataframes
overall_mean = pd.DataFrame()
list_a = [] 
for i in range(len(animate_mean)):
    list_a.append(animate_mean.iloc[i] - inanimate_mean.iloc[i])    

overall_mean['mean'] = list_a


def get_mean():
    std_overall = np.std(list_a)
    mean_overall = overall_mean.mean()    
    N = 44
    t = mean_overall/ (std_overall/ sqrt(N)) # type: ignore     
    return t
    


# Create a bar plot of average voxel response for animate and inanimate objects.
# Needs to have a error bar added to it to indicate the standard error of the mean

# Standard error of the mean is neccessary instead of the standard deveiation

animate_mean = animate_data.mean(axis=1)
inanimate_mean = inanimate_data.mean(axis=1)

animate_sem = animate_mean.sem()
inanimate_sem = inanimate_mean.sem()

#============================ Plots ===================================#
def plot1():
        
    plt.bar(1,animate_mean.mean(),yerr= animate_sem.mean(),align="center",alpha=0.7,ecolor="red",capsize=10)
    plt.bar(2,inanimate_mean.mean(),yerr=inanimate_sem.mean(),align="center",alpha=0.7,ecolor="red",capsize=10)
    plt.xticks([1,2],["animate","inanimate"])
    plt.axhline(y=0,color="black")

    plt.ylabel("Response Amplitude")
    plt.xlabel("Average Response Amplitude")
    plt.show()

def plot2():
    plt.bar(a,amplitude,align="center",alpha=0.5,ecolor="black",capsize=10)
    plt.axhline(y=0,color="black")
    plt.xlabel("animate - inanimate")
    plt.ylabel("Response amplitude")
    plt.show()


#============================= Part 2 ===================================#
df_data = pd.read_csv(open(file_path("NeuralResponses_S2.txt"),"r"),sep=",")

df_pt2 = pd.DataFrame(data=df_data)
# Create a list with 22 1's and 22 -1's
labels = []
for i in range(44):
    labels.append(1)
for i in range(44):
    labels.append(-1)
    
labels = np.array(labels)
# Reomves the first 12 columns from the dataframe
df_pt2 = df_pt2.iloc[:,12:]


# append animate_labels to the df2 dataframe
animate_labels = pd.DataFrame(labels,columns=["labels"])
data = pd.concat([df_pt2, animate_labels], axis=1)

#==========================================SVM========================================================#

# Splits the data into a training and test dataset
training_dataset, test_dataset = train_test_split(data,test_size=0.5, random_state=42)


# Returns the labels and the data from the dataset where the labels are the last column in the dataset
def get_labels_and_data(dataset):
    labels = dataset.iloc[:,-1]
    data = dataset.iloc[:,:-1]
    return labels, data

y,x = get_labels_and_data(training_dataset)
clf = svm.SVC(kernel="linear")
clf.fit(x,y)


# Create a confusion matrix to see how well the model performed
y_true = test_dataset.iloc[:,-1]
y_pred = clf.predict(test_dataset.iloc[:,:-1])
# Create a plot confusion matrix to see how well the model performed


# Gets the first 20 weights from the model.
weights = clf.coef_

weights = weights[0]
weights = weights[0:20]



# create a plot to compare the weights and amplitudes
def plot3():
    plt.scatter(weights,amplitude)
    plt.xlabel("weights")
    plt.ylabel("amplitude")
    plt.show()

coeff = np.corrcoef(weights,amplitude)


#====================Human vs Non-human======================#

# Gets the human and non human values from the dataframe containing all the data (Categories + voxel responses)
human = df_full.loc[df_full["human"] == "1"]
human.drop(human.tail(4).index, inplace=True)
print(human)
nonhuman = df_full.loc[df_full["nonhumani"] == "1"]

human_df = pd.concat([human,nonhuman])
human_df = human_df.iloc[:,12:]
print(human_df)
human_df = human_df.reset_index(drop=True)


# Creates a list that will contain the labels on the

b = [1 if x <20 else -1 for x in range(40)]

# Creates a np array from the list created in the loop        
b= np.array(b)
labels2 = pd.DataFrame(b, columns=["labels"])
split_ds = pd.concat([human_df,labels2],axis=1)
print(split_ds)

train_data2, test_data2= train_test_split(split_ds,test_size=0.5,random_state=47)
print(train_data2)

y2,x2 = get_labels_and_data(train_data2)

clf2 = svm.SVC(kernel="linear")
clf2.fit(x2,y2)

y_true = test_data2.iloc[:,-1]
y_pred = clf2.predict(test_data2.iloc[:,:-1])
# Create a plot confusion matrix to see how well the model performed


# Gets the first 20 weights from the model.
weights2 = clf2.coef_

weights2 = weights2[0]
weights2 = weights2[0:20]



# create a plot to compare the weights and amplitudes
def plot4():
    plt.scatter(weights2,amplitude)
    plt.xlabel("weights")
    plt.ylabel("amplitude")
    plt.show()

coeff = np.corrcoef(weights2,amplitude)

