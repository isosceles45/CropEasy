from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('FinalCP (1).csv')

features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
#features = df[['temperature', 'humidity', 'ph', 'rainfall']]
labels = df['label']

# Initialzing empty lists to append all model's name and corresponding name
acc = []
model = []
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)

####Importing the actual model

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain,Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('RF')
print("RF's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))

import pickle
# Dump the trained Naive Bayes classifier with Pickle
RF_pkl_filename = 'randomforrest2.pkl'
# Open the file to save as pkl file
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
# Close the pickle instances
#RF_Model_pkl.close()
pickle.dump(RF,open('randomforrest2.pkl','rb'))
model=pickle.load(open('randomforrest2.pkl','rb'))

######Making an actual pediction

##data = np.array([[104,18, 30, 23.603016, 60.3, 6.7, 140.91]])
prediction = RF.predict(data)
##print(prediction)
##
##data = np.array([[120,18, 30, 23.603016, 75.3, 6.7, 155.91]])
##prediction = RF.predict(data)
##print(prediction)#