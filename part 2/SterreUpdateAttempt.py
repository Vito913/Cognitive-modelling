import pandas as pd
import numpy as np
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
#==========================================Data Pre Processing========================================================#

## Get the path of the file
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
df2 = pd.read_csv(open(file_path("NeuralResponses_S2.txt"),"r"), sep=",")

# Adds the two dataframes together
df = pd.concat([df, df2], axis=1)

# Create a list with 22 1's and 22 -1's
labels = []
for i in range(44):
    labels.append(1)
for i in range(44):
    labels.append(-1)
    
# Removes the first 12 columns from the dataframe
df = df.iloc[:,12:]

# append animate_labels to the df2 dataframe
animate_labels = pd.DataFrame(labels,columns=["labels"])
data = pd.concat([df, animate_labels], axis=1)

# Splits the data into a training and test dataset
training_dataset, test_dataset = train_test_split(data,test_size=0.5, random_state=42)
print(training_dataset)

# Returns the labels and the data from the dataset where the labels are the last column in the dataset
def get_labels_and_data(dataset):
    labels = dataset.iloc[:,-1]
    data = dataset.iloc[:,:-1]
    return labels, data

#==========================================SVM========================================================#

y,x = get_labels_and_data(training_dataset)

clf = svm.SVC()
clf.fit(x,y)
clf.predict(test_dataset.iloc[:,:-1])

# Create a confusion matrix to see how well the model performed
y_true = test_dataset.iloc[:,-1]
y_pred = clf.predict(test_dataset.iloc[:,:-1])
print(confusion_matrix(y_true, y_pred))

# Create a plot confusion matrix to see how well the model performed
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf, test_dataset.iloc[:,:-1], test_dataset.iloc[:,-1],
                                 display_labels=['animate','inanimate'],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
plt.show()

y_true_pearson = np.corrcoef(y_true)
y_pred_pearson = np.corrcoef(y_pred)

print(y_true_pearson)
print(y_pred_pearson)

##
##
##
# Please see my attempt at the human vs nonhuman stuff below
##
##
##

# Get the data from the categoryVectors file and append it to a dataframe
df3 = pd.DataFrame(data= vector_data)

# separate the data in the dataframe by the "," characters
df3 = df3[0].str.split(",", expand=True)

# Remove the "\n" character from the last column
df3[11] = df3[11].str.replace("\n", "")

# removes the first row of the dataframe
df3 = df3.iloc[1:]

df3.index = df3.index -1

# adds the column names to the dataframe
df3.columns = category_vectors

# load a different dataframe containing data from NeuralResponses.txt
df4 = pd.read_csv(open(file_path("NeuralResponses_S2.txt"),"r"), sep=",")

# Adds the two dataframes together
df3 = pd.concat([df3, df4], axis=1)

# Create a list with 22 1's and 22 -1's
labels = []
for i in range(44):
    labels.append(1)
for i in range(44):
    labels.append(-1)
    
# Removes the first 10 and last 2 columns from the dataframe
df3 = df3.iloc[:,10:2] # <= DOES THIS WORK? I can't test it :/

# append animate_labels to the df2 dataframe
animate_labels = pd.DataFrame(labels,columns=["labels"])
data = pd.concat([df3, animate_labels], axis=1)

# Splits the data into a training and test dataset
training_dataset, test_dataset = train_test_split(data,test_size=0.5, random_state=42)
print(training_dataset)
