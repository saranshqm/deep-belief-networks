# deep-belief-networks


``` python
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.metrics import confusion_matrix


# np.random.seed(1337)


from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score, precision_score,recall_score, f1_score


from sklearn.preprocessing import MinMaxScaler as Scaler
import pandas as pd
f = 1




# use your training file
train = pd.read_csv('train.csv')


from scipy import stats
import numpy as np

#<<<<<<<<<<<<<<<<<----Removing The outliers---------------->>>>>>>>>>>>>>>>>>>>>>>>
z = np.abs(stats.zscore(train))
threshold = 3
original_train = train
train = train[(z < 3).all(axis=1)]
#<<<<<<<<<<<<<<<<<--------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>

Y= train['Disease Status (NSCLC: primary tumors; Normal: non-tumor lung tissues)']
X = train[train.columns[:-1]]



#<<<<<<<<<<--------------------Oversampling--------->>>>>>>>>>>>
sm = SMOTE(random_state=42)
X, Y = sm.fit_sample(X, Y)
# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

#<--------------scaling the inputs-------------------->
scaler = Scaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


#<<<<<<<<<<--------------------Oversampling--------->>>>>>>>>>>>
sm = SMOTE(random_state=42)
X_train, Y_train = sm.fit_sample(X_train, Y_train)
#<<<<<<<<<<------------------------------------------>>>>>>>>>>>
print(len(X_train))
# Training

# specify your parameters


#<<<<<<<<<<<<<<<------SupervisedDBNClassification is used for classification-------------->>>>>>>>>>>>>>
#<<<<<<<<<<<<<<<------SupervisedDBNRegression is used for regression---------------------->>>>>>>>>>>>>>

classifier = SupervisedDBNClassification(hidden_layers_structure=[10,8],
                                         learning_rate_rbm=0.01,
                                         learning_rate=0.1 ,
                                         n_epochs_rbm= 1, 
                                         n_iter_backprop=20000,
                                         batch_size=1000,
                                         activation_function='relu',
                                         dropout_p=0.5)


classifier.fit(X_train, Y_train)


# Test
Y_pred = classifier.predict(X_test)
a = accuracy_score(Y_test, Y_pred)
print('Done.\nAccuracy: %f' % a)
print('Done.\nPrecision: %f' % precision_score(Y_test, Y_pred))
print('Done.\nRecall: %f' % recall_score(Y_test, Y_pred))
print('Done.\nf1 score: %f' % f1_score(Y_test, Y_pred))
#print('Done.\nf1 score: %f' % classification_report(Y_test, Y_pred))
#print('Done.\nf1 score: %f' % confusion_matrix(Y_test, Y_pred))
cm1 = confusion_matrix(Y_test, Y_pred)
print(cm1)

total1=sum(sum(cm1))
accuracy1=(cm1[0,0]+1+cm1[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)
#precision_score


from sklearn.metrics import cohen_kappa_score

print('kappa:-',cohen_kappa_score(Y_test, Y_pred))
from sklearn.metrics import roc_auc_score

print('validation roc score',roc_auc_score(Y_pred,Y_test))


test = pd.read_csv('test.csv')

```
