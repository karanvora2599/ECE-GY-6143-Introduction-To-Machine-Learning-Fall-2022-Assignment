from scipy.io import loadmat
from sklearn.model_selection import train_test_split

Problem1Data = loadmat(r"/home/karanvora/Documents/New York University/Classes/Semester 1/Machine Learning/Assignments/Assignment 1/problem1.mat")
X_Data = []
Y_Data = []

for value in range(len(Problem1Data['x'])):
    X_Data.append(Problem1Data['x'][value][0])
    Y_Data.append(Problem1Data['y'][value][0])

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_Data, Y_Data, test_size=0.2)

print(len(X_Train))
print(len(X_Test))