import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  plot_confusion_matrix ,classification_report
import warnings
warnings.filterwarnings('ignore')

X_train = pd.read_csv('../input/kepler-labelled-time-series-data/exoTrain.csv')
graphical_analysis_df = pd.read_csv('../input/kepler-labelled-time-series-data/exoTrain.csv')
y_train = X_train['LABEL']
X_train = X_train.drop(columns=['LABEL'], axis=1)
X_test = pd.read_csv('../input/kepler-labelled-time-series-data/exoTest.csv')
y_test = X_test['LABEL']
X_test = X_test.drop(columns=['LABEL'], axis=1)
X_train.head()

def flux_graph(dataset, row, dataframe, planet):
    if dataframe:
        fig = plt.figure(figsize=(20,5))
        ax = fig.add_subplot()
        ax.set_title(planet, color='black', fontsize=22)
        ax.set_xlabel('time', color='black', fontsize=18)
        ax.set_ylabel('flux_' + str(row), color='black', fontsize=18)
        ax.grid(False)
        flux_time = list(dataset.columns)
        flux_values = dataset[flux_time].iloc[row]
        ax.plot([i + 1 for i in range(dataset.shape[1])], flux_values, 'blue')
        ax.tick_params(colors = 'black', labelcolor='black', labelsize=14)
        plt.show()
    else:
        fig = plt.figure(figsize=(20,5))
        ax = fig.add_subplot()
        
        ax.set_title(planet, color='black', fontsize=22)
        ax.set_xlabel('time', color='black', fontsize=18)
        ax.set_ylabel('flux_' + str(row), color='white', fontsize=18)
        ax.grid(False)
        flux_values = dataset[row]
        ax.plot([i + 1 for i in range(dataset.shape[1])], flux_values, 'blue')
        ax.tick_params(colors = 'black', labelcolor='black', labelsize=14)
        plt.show()
def show_graph(dataframe, dataset):
    with_planet = graphical_analysis_df[graphical_analysis_df['LABEL'] == 2].head(3).index
    wo_planet = graphical_analysis_df[graphical_analysis_df['LABEL'] == 1].head(3).index

    for row in with_planet:
        flux_graph(dataset, row, dataframe, planet = 'transiting planet')
    for row in wo_planet:
        flux_graph(dataset, row, dataframe, planet = 'no transiting planet')
show_graph(True, dataset = graphical_analysis_df.loc[:, graphical_analysis_df.columns != 'LABEL'])

scl = StandardScaler()
scl.fit(X_train)
X_train_scl = scl.transform(X_train)
scl.fit(X_test)
X_test_scl = scl.transform(X_test)
DT = DecisionTreeClassifier()
DT.fit(X_train_scl, y_train)
prediction_DT=DT.predict(X_test_scl)
train_score_DT = DT.score(X_train_scl, y_train)
test_score_DT = DT.score(X_test_scl, y_test)
print(f"Decision Tree train score: {train_score_DT}")
print(f"Decision Tree test score: {test_score_DT}")
print('Decision Tree Classifier')
print(classification_report(y_test, prediction_DT))
plot_confusion_matrix(DT,X_test_scl,y_test)
plt.show()
RF = RandomForestClassifier()
RF.fit(X_train_scl, y_train)
prediction_RF=RF.predict(X_test_scl)
train_score_RF = RF.score(X_train_scl, y_train)
test_score_RF = RF.score(X_test_scl, y_test)
print(f"RF train score: {train_score_RF}")
print(f"RF test score: {test_score_RF}")
print(classification_report(y_test, prediction_RF))
plot_confusion_matrix(RF,X_test_scl,y_test)
plt.show()
LR = LogisticRegression()
LR.fit(X_train_scl,y_train)
prediction_LR=LR.predict(X_test_scl)
train_score_LR = LR.score(X_train_scl, y_train)
test_score_LR = LR.score(X_test_scl, y_test)
print(f"LR train score: {train_score_LR}")
print(f"LR test score: {test_score_LR}")
print(classification_report(y_test, prediction_LR))
plot_confusion_matrix(LR,X_test_scl,y_test)
plt.show()
data = {'Decision Tree': {'Train': train_score_DT, 'Test': test_score_DT},
        'Random Forest': {'Train': train_score_RF, 'Test': test_score_RF},
       'Logistic Regression': {'Train': train_score_LR, 'Test': test_score_LR}}
df = pd.DataFrame(data)
df = df.T
df ['sum'] = df.sum(axis=1)
df.sort_values('sum', ascending=False)[['Test','Train']].plot.bar() 
plt.ylabel('Score')
