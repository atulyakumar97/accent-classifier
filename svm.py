import pandas as pd

print('Importing CSV')
data=pd.read_csv('Features.csv', low_memory=False)

X=data.iloc[:,1:6767] #feature values
y=data.iloc[:,6767] #label

del(data)

print('Encoding Categorical Data')
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y.ravel())

print('Splitting the dataset into the Training set and Test set')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.0125, random_state = 1,shuffle=True)

# -----------------------------------SVM Grid Search -------------------------------------------------#
#from sklearn.svm import SVC
#best_score = 0  
#best_params = {'C': None, 'gamma': None}
#
#C_values = [0.01]  # 0.03, 0.1, 0.3, 1,3, 10, 30, 100
#gamma_values = [0.01, 3, 10, 30, 100, 0.03, 0.1, 0.3, 1] 
#
#print('SVM grid search')
#for C in C_values:  
#    for gamma in gamma_values:
#        print(C,gamma)
#        classifier = SVC(C=C, gamma=gamma,kernel = 'rbf', random_state = 1,max_iter=1000000,class_weight='balanced')
#        classifier.fit(X_train, y_train)
#        score = classifier.score(X_test, y_test)
#        print(score*100)
#        
#        if score > best_score:
#            best_score = score
#            best_params['C'] = C
#            best_params['gamma'] = gamma
#
#print(best_score, best_params)
#
#classifier = SVC(C=best_params['C'], gamma=best_params['gamma'],kernel = 'rbf', random_state = 1,max_iter=1000000,class_weight='balanced')
#classifier.fit(X_train, y_train)
#score = classifier.score(X_test, y_test)
#print(score*100)
#-----------------------------------------------------------------------------------------------------

print('Fitting Kernel SVM to the Training set')
from sklearn.svm import SVC
classifier = SVC(C=2,gamma=3,kernel = 'rbf', random_state =1,max_iter=100000)
classifier.fit(X_train, y_train)

score = classifier.score(X_test,y_test)
print('Score = ',score*100)


import pickle

# save the model to disk
filename = 'finalised_model.sav'
pickle.dump(classifier, open(filename, 'wb'))
 
## some time later...
# 
## load the model from disk
#loaded_classifier = pickle.load(open(filename, 'rb'))
#result = loaded_classifier.score(X_test, y_test)
#print(result)