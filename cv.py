#import all necessary tools
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.preprocessing import MinMaxScaler,label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold,cross_val_score
from sklearn.metrics import accuracy_score,roc_curve,auc
from sklearn import metrics
from itertools import cycle
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
#Defined classifiers
clsr_names=["Decision Tree", "Random Forest", "Logistic Regression"]

classifiers = [DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    LogisticRegression()]
#Read data in DataFrame
df01=pd.read_csv('adult.data', sep=',',header=0)
#Defined the column names
df01.columns= ["age", "workclass", "fnlwgt", "education", "education_num",
  "marital_status", "occupation", "relationship", "race", "gender",
  "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_bracket"]
df02=df01.dropna(axis=1, how='all')
df=df02.dropna(axis=0, how='any')
cols=df.dtypes
colnms=df.columns
#Check data types for all columns
print(cols)
print(colnms)
#Check which variables are character and need to processed by One Hot Encoding
i=0
cat_cols=[]
for eachcol in cols:
    if eachcol.name=="object":
        cat_cols.append(colnms[i])
    i+=1
print(cat_cols)
#Encod all character variables
df1=pd.get_dummies(df,columns=cat_cols)
n=len(df1.index)
m=len(df1.columns)
print(df1.columns)
print("n-index= "+str(n))
print("m-columns= "+str(m))
#Check the shape of matrix which will apply to build the ML models
print(df1.shape)
#Select predictors and labeled data for modeling
x_all=df1.iloc[:,0:(m-2)]
y_all=df1.iloc[:,-1]
#Creat a new file to include predictions and CV scores
f=open('top_vars.xls','w')
#Evaluate the performance of each ML classifiers by using cross validation
clf = classifiers[0]
f.write("Decision Tree:\n")
print(cross_val_score(clf,x_all,y_all,scoring='accuracy',cv=10))
accuracy=cross_val_score(clf,x_all,y_all,scoring='accuracy',cv=10).mean()*100
f.write("CV accuracy score = {0:.3f}\n".format(accuracy))

clf = classifiers[1]
f.write("Random Forest:\n")
print(cross_val_score(clf,x_all,y_all,scoring='accuracy',cv=10))
accuracy=cross_val_score(clf,x_all,y_all,scoring='accuracy',cv=10).mean()*100
f.write("CV accuracy score = {0:.3f}\n".format(accuracy))

clf = classifiers[2]
f.write("Logistic Regression:\n")
print(cross_val_score(clf,x_all,y_all,scoring='accuracy',cv=10))
accuracy=cross_val_score(clf,x_all,y_all,scoring='accuracy',cv=10).mean()*100
f.write("CV accuracy score = {0:.3f}\n".format(accuracy))
#Split off test and train data and normalize data
x_trn, x_tst, y_trn, y_tst = train_test_split(x_all, y_all, test_size=0.4, random_state=42)
print(x_trn.shape)
scaler = MinMaxScaler()
scaler.fit(x_trn)
x_trn_n=scaler.transform(x_trn)
x_tst_n=scaler.transform(x_tst)
#Build the model and predict the parameters for Desicion Tree and calculate the weights of top10 features contributing to income greater than 50k on the screen
clf = classifiers[0]
model=clf.fit(x_trn_n,y_trn)
imp1=model.feature_importances_
var2imp1=dict(zip(list(df1),imp1))
var2imp1_sorted=pd.DataFrame(columns=['variable','weight'])
for key in sorted(var2imp1, key=lambda k:abs(var2imp1[k]),reverse=True):
    temp=pd.DataFrame([[key,var2imp1[key]]],columns=['variable','weight'])
    var2imp1_sorted=var2imp1_sorted.append(temp)
print("Top 10 important variables-Decision Tree:")
print(var2imp1_sorted[0:10])
f.write("Top 10 Weighted Variables - Decision Tree:"+"\n")
f.write("Rank\tVariable\tWeight\n")
for g in range (0,10,1):
    f.write(str(g+1)+'\t'+str(var2imp1_sorted.iloc[g,0])+'\t'+str(var2imp1_sorted.iloc[g,1])+"\n")
#Visualize the  prediction of Decision Tree by bar chart and save in file named "plot.pdf"
var1_names=list(var2imp1_sorted['variable'][0:10])
var1_imp=list(var2imp1_sorted['weight'][0:10])
y_pos = np.arange(len(var1_names),0,-1)
fig = plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.barh(y_pos, var1_imp,color='#87CEEB', align='center', alpha=0.5)
plt.yticks(y_pos, var1_names)
plt.xlabel('Weight')
plt.title('Decision Tree')
plt.ylim(0,11)
#Build the model and predict the parameters for Random Forest and calculate the weights of top10 features contributing to income greater than 50k on the screen
clf = classifiers[1]
model=clf.fit(x_trn_n,y_trn)
imp2=model.feature_importances_
var2imp2=dict(zip(list(df1),imp2))
var2imp2_sorted=pd.DataFrame(columns=['variable','weight'])
for key in sorted(var2imp2, key=lambda k:abs(var2imp2[k]),reverse=True):
    temp=pd.DataFrame([[key,var2imp2[key]]],columns=['variable','weight'])
    var2imp2_sorted=var2imp2_sorted.append(temp)
print("Top 10 important variables-Random Forest:")
print(var2imp2_sorted[0:10])
f.write("Top 10 Weighted Variables - Random Forest:"+"\n")
f.write("Rank\tVariable\tWeight\n")
for j in range (0,10,1):
    f.write(str(j+1)+'\t'+str(var2imp2_sorted.iloc[j,0])+'\t'+str(var2imp2_sorted.iloc[j,1])+"\n")
#Visualize the  prediction of Random Forest by bar chart and save in file named "plot.pdf"
var2_names=list(var2imp2_sorted['variable'][0:10])
var2_imp=list(var2imp2_sorted['weight'][0:10])
y_pos = np.arange(len(var2_names),0,-1)
plt.subplot(2, 2, 2)
plt.barh(y_pos, var2_imp,color='#87CEEB', align='center', alpha=0.5)
plt.yticks(y_pos, var2_names)
plt.xlabel('Weight')
plt.title('Random Forest')
plt.ylim(0,11)
#Build the model and predict the parameters for Logistic Regression and calculate the weights of top10 features contributing to income greater than 50k on the screen
clf = classifiers[2]
model=clf.fit(x_trn_n,y_trn)
weight=model.coef_[0]
var2wgt=pd.DataFrame(list(zip(list(df1),weight)),columns=['variable','weight'])
var2wgt_sorted=var2wgt.reindex(var2wgt.weight.abs().sort_values(ascending=False).index)
f.write("Top 10 Weighted Variables - Logistic Regression:"+"\n")
f.write("Rank\tVariable\tWeight\n")
for i in range (0,10,1):
    f.write(str(i+1)+'\t'+str(var2wgt_sorted.iloc[i,0])+'\t'+str(var2wgt_sorted.iloc[i,1])+"\n")
#Visualize the  prediction of Logistic Regression by bar chart and save in file named "plot.pdf""
var3_names=list(var2wgt_sorted['variable'][0:10])
var3_imp=list(var2wgt_sorted['weight'][0:10].abs())
y_pos = np.arange(len(var3_names),0,-1)
plt.subplot(2, 2, 3)
plt.barh(y_pos, var3_imp,color='#87CEEB',align='center',alpha=0.5)
plt.yticks(y_pos, var3_names)
plt.xlabel('Weights')
plt.ylabel('Top 10 Features')
plt.title('Logistic Regression ')
plt.ylim(0,11)
plt.tight_layout()
fig.savefig('plot.pdf',dpi=400)
f.close()
