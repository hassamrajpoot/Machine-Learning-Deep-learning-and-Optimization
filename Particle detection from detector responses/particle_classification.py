import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../input/pid-5M.csv')

df.head()

df.describe()

df.isnull().sum()

df.corr()

X = df.drop('id', axis=1)
y = df['id']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

ADC = AdaBoostClassifier()
ADC.fit(x_train, y_train)
pred_ADC = ADC.predict(x_test)
print('accuracy score:', accuracy_score(y_test, pred_ADC))

XG = XGBClassifier()
XG.fit(x_train, y_train)
pred_XG = XG.predict(x_test)
print('accuracy score:', accuracy_score(y_test, pred_XG))

RF = RandomForestClassifier()
RF.fit(x_train, y_train)
pred_RF = RF.predict(x_test)
print('accuracy score:', accuracy_score(y_test, pred_RF))

LR = LogisticRegression()
LR.fit(x_train, y_train)
pred_LR = LR.predict(x_test)
print('accuracy score:', accuracy_score(y_test, pred_LR))

data= {'XGBoost': {'Train': XG.score(x_train, y_train), 'Test': XG.score(x_test, y_test)},
        'Random Forest': {'Train': RF.score(x_train, y_train), 'Test': RF.score(x_test, y_test)},
       'Logistic Regression': {'Train': LR.score(x_train, y_train), 'Test': DT.score(x_test, y_test)},
      'Adaboost classifier': {'Train': ADC.score(x_train, y_train), 'Test': ADC.score(x_test, y_test)},
      }
df = pd.DataFrame(data)
df = df.T
df ['sum'] = df.sum(axis=1)
df.sort_values('sum', ascending=False)[['Test','Train']].plot.bar() 
plt.ylabel('Score')
