#import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")


data=pd.read_csv("weather.csv")
dummy=pd.get_dummies(data['RainToday'])
data2=pd.concat((data,dummy),axis=1)
data2=data2.drop(['RainToday'],axis=1)
data2=data2.drop(['No'],axis=1)
data2=data2.rename(columns={'Yes':'RainToday'})
data2.head()
dy2=pd.get_dummies(data2['RainTomorrow'])
data2=pd.concat((data2,dy2),axis=1)
data2=data2.drop(['RainTomorrow'],axis=1)
data2=data2.drop(['No'],axis=1)
data2=data2.rename(columns={'Yes':'RainTommorow'})
X_axis=data2[['MinTemp','MaxTemp','Rainfall']]
Y_axis=data2[['RainTommorow']]
X_axis.head()
X_train, X_test, y_train, y_test = train_test_split(X_axis, Y_axis, test_size=0.3, random_state=0)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
#score = log_reg.score(X_test, y_test)
#print(score)
# Saving model to disk
pickle.dump(log_reg, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))