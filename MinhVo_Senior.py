# Minh Vo
# CS488 - Senior Capstone Experience
# MinhVo_Senior.py

import glob, os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import linear_model
from sklearn import tree
from sklearn.cross_validation import train_test_split

# Create a list of the csv files
all_files = glob.glob(os.path.join('', "*.csv"))
file_list = []

for f in all_files:
    # Read the csv files using pandas
    frame = pd.read_csv(f, index_col = None, header = 0, 
                        error_bad_lines = False, warn_bad_lines = False)
    # error_bad_lines set to False to skip lines that have too many fields (commas)
    file_list.append(frame)

# Create the pandas data frame from the read csv files
df = pd.DataFrame()
df = pd.concat(file_list)

# Include only the rows in which the HomeTeam column is not blank
# This is an error found in the data
df = df[df['HomeTeam'].notnull()]

# Create another data frame from the original data
# This data contains only the columns of the best features and the result
data = df[['HomeTeam', 'AwayTeam', 'HTHG', 'HTAG', 'HST', 'AST', 'FTR']]

# Take all columns except for 'FTR'
x_all = data.drop(['FTR'], 1)

# Take only 'FTR' column
y_all = data['FTR']

# Convert the categorical variables into integers
le = preprocessing.LabelEncoder()
x_all = x_all.apply(le.fit_transform)

# Turn the possible values into binary features
# Only one of them is active (1)
enc = preprocessing.OneHotEncoder()
x_all = enc.fit_transform(x_all).toarray()

# Split the data into training and testing set
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, 
    test_size = 0.10, random_state = 42)

def accuracy(algorithm, x_train, y_train, x_test, y_test):
    """Function to print the accuracy of the algorithm.
    @params:
    algorithm: the machine learning algorithm
    x_train: the features for training
    y_train: the labels for training
    x_test: the features for testing
    y_test: the labels for testing
    """
    # Fit the model
    model = algorithm.fit(x_train, y_train)
    # Print the accuracy
    print(model.score(x_test, y_test))

def main():
    # The four algorithms to be used
    gnb = GaussianNB()
    decisionTree = tree.DecisionTreeClassifier()
    svc = svm.SVC()
    logisticReg = linear_model.LogisticRegression()
    
    # Create a list of the algorithms
    algorithms = [gnb, decisionTree, svc, logisticReg]
    
    # Run the accuracy function with the list of 4 algorithms
    for algo in algorithms:
        accuracy(algo, x_train, y_train, x_test, y_test)
        
if __name__ == "__main__":
    main()