import pandas as pd
import numpy as np
import os
from math import *
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xgboost as xgb
from sklearn import preprocessing, cluster
from sklearn import metrics
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

detectors_data = pd.read_csv('../input/detectors.zip')
detectors_data.head()

sample_submission_data = pd.read_csv('../input/sample_submission.csv.zip')
sample_submission_data.head()

train_filenames = os.listdir('../input/train_1')
train_event_ids = np.unique(sorted([lambda filename: int(filename[5:].split('-')[0]) for filename in train_filenames]))
print('train event ids:', train_event_ids[:10],'...', train_event_ids[-10:])

def get_by_event_id_train(id_):
    if id_ in train_event_ids:
        return sorted(np.array(train_filenames)[[id_ == lambda filename: int(filename[5:].split('-')[0]) for filename in train_filenames]])

cells_data = pd.read_csv('../input/train_1/'+get_by_event_id_train(1000)[0]) 
hits_data = pd.read_csv('../input/train_1/'+get_by_event_id_train(1000)[1])
particles_data = pd.read_csv('../input/train_1/'+get_by_event_id_train(1000)[2])
truth_data = pd.read_csv('../input/train_1/'+get_by_event_id_train(1000)[3])

cells_data.head()

hits_data.head()

particles_data.head()

truth_data.head()

test_filenames = os.listdir('../input/test/')
test_event_ids = np.unique(sorted([lambda filename: int(filename[5:].split('-')[0]) for filename in test_filenames]))
print('test event ids:', test_event_ids[:10],'...', test_event_ids[-10:])

def get_by_event_id_test(id_):
    if id_ in test_event_ids:
        return sorted(np.array(test_filenames)[[id_ == lambda filename: int(filename[5:].split('-')[0]) for filename in test_filenames]])

test_cells_data = pd.read_csv('../input/test/' + get_by_event_id_test(10)[0])
test_hits_data = pd.read_csv('../input/test/' + get_by_event_id_test(10)[1])

test_cells_data.head()

test_hits_data.head()

hits_data.index = hits_data.hit_id
hits_data = hits_data.drop('hit_id', axis=1)

print('Len:', len(cells_df))
cells_df.head()

print('Len:', len(hits_data))
hits_data.head()

truth_data.head()

sample_particle_id = 0
while sample_particle_id == 0:
    sample_particle_id = int(truth_data.sample()['particle_id'])
print(sample_particle_id)

hits_flow = np.array(truth_data[truth_data['particle_id']==sample_particle_id]['hit_id'])

data = np.array(truth_data[truth_data['particle_id']==sample_particle_id][['tx', 'ty', 'tz']])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2] , c='red', marker='o', edgecolor='k')
ax.plot(data[:,0], data[:,1], data[:,2] , '-', lw=3, alpha=0.4)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
fig.tight_layout()
plt.show()

#hits_data.loc[hits_flow]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
data = np.array(hits_data.loc[hits_flow][['x', 'y', 'z']])
ax.scatter(data[:,0], data[:,1], data[:,2] , c='red', marker='o', edgecolor='k', s=np.ones(len(data))*50)
ax.plot(data[:,0], data[:,1], data[:,2] , 'k-', lw=3, alpha=0.4)
data = np.array(hits_df[['x', 'y', 'z']].sample(100))
ax.scatter(data[:,0], data[:,1], data[:,2] , c='blue', marker='o', edgecolor='k')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.view_init(elev=20)
fig.tight_layout()
plt.show()

fig = plt.figure(figsize=(15, 10))
elev_range = [int(np.random.random()*90) for i in range(0, 3)]
azim_range = [int(np.random.random()*180) for i in range(0, 3)]
for i, elev_angle in enumerate(elev_range):
    for j, azim_angle in enumerate(azim_range):
        ax = fig.add_subplot(331 + i*3 + j, projection='3d')
        data = np.array(hits_data.loc[hits_flow][['x', 'y', 'z']])
        ax.scatter(data[:,0], data[:,1], data[:,2] , c='red', marker='o', edgecolor='k', s=np.ones(len(data))*50)
        ax.plot(data[:,0], data[:,1], data[:,2] , 'k-', lw=3, alpha=0.4)
        data = np.array(hits_data[['x', 'y', 'z']].sample(200))
        ax.scatter(data[:,0], data[:,1], data[:,2] , c='blue', marker='o', edgecolor='k', alpha=0.3)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.view_init(elev=elev_angle, azim=azim_angle)
    
fig.tight_layout()
plt.show()

hits_data.head()

cells_data.index = cells_data.hit_id
cells_data = cells_data.drop('hit_id', axis=1)

cells_data.join(hits_data).head()

truth_data.index = truth_data.hit_id
truth_data = truth_data.drop('hit_id', axis=1)

global_data = cells_data.join(hits_data).join(truth_data)

global_data.sort_values('particle_id').head(10)

global_data[['value', 'weight']].corr()

fig = plt.figure(figsize=(15, 10))
elev_range = [int(np.random.random()*90) for i in range(0, 3)]
azim_range = [int(np.random.random()*180) for i in range(0, 3)]
for i, elev_angle in enumerate(elev_range):
    for j, azim_angle in enumerate(azim_range):
        ax = fig.add_subplot(331 + i*3 + j, projection='3d')
        data = np.array(global_data[global_data['weight']!=0][['x', 'y', 'z']])[:200]
        ax.scatter(data[:,0], data[:,1], data[:,2] , c='red', marker='o', edgecolor='k', s=np.ones(len(data))*50)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.view_init(elev=elev_angle, azim=azim_angle)
fig.tight_layout()
plt.show()

sample_particle = int(global_data[global_data['weight']!=0].sample()['particle_id'])

global_data[global_data['particle_id'] == sample_particle].head()

scl = preprocessing.StandardScaler()
dbscan = cluster.DBSCAN(eps=0.0076, min_samples=1, algorithm='kd_tree', n_jobs=-1)

x = hits_data.x.values
y = hits_data.y.values
z = hits_data.z.values
r = np.sqrt(x**2+y**2+z**2)
x2 = x/r
y2 = y/r
r2 = np.sqrt(x**2+y**2)
z2 = z/r2

def get_features(dataframe, theta=0):
    x = dataframe.x.values
    y = dataframe.y.values
    z = dataframe.z.values
    r = np.sqrt(x**2+y**2+z**2)
    x2 = x/r
    y2 = y/r
    r2 = np.sqrt(x**2+y**2)
    z2 = z/r2
    dataframe['x2'] = x2
    dataframe['y2'] = y2
    dataframe['z2'] = z2
    dataframe['r'] = r2
    return dataframe

transformed_hits_data = get_features(hits_data)

fig = plt.figure(figsize=(15, 10))
sub_sample = transformed_hits_data.sample(4000)
ax = fig.add_subplot(121, projection='3d')
data = np.array(sub_sample[['x', 'y', 'z']])
ax.scatter(data[:,0], data[:,1], data[:,2] , c='red', marker='o', edgecolor='k', s=np.ones(len(data))*50)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.view_init(elev=20, azim=0)
ax = fig.add_subplot(122, projection='3d')
data = np.array(sub_sample[['x2', 'y2', 'z2']])
ax.scatter(data[:,0], data[:,1], data[:,2] , c='red', marker='o', edgecolor='k', s=np.ones(len(data))*50)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.view_init(elev=20, azim=0)  
fig.tight_layout()
plt.show()

sample_particle_id = 0
while sample_particle_id == 0:
    sample_particle_id = int(truth_data.sample()['particle_id'])
print('Choice of a random particle: {}'.format(sample_particle_id))
print('Transformed 3D DF:')
display(transformed_hits_data.loc[list(truth_data[truth_data['particle_id']==sample_particle_id].index)].head())
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(121, projection='3d')
sub_sample = transformed_hits_data.loc[list(truth_data[truth_data['particle_id']==sample_particle_id].index)]
data = np.array(sub_sample[['x', 'y', 'z']])
ax.scatter(data[:,0], data[:,1], data[:,2] , c='red', marker='o', edgecolor='k', s=np.ones(len(data))*50, label='Same track particles')
sub_sample = transformed_hits_data.sample(100)
data = np.array(sub_sample[['x', 'y', 'z']])
ax.scatter(data[:,0], data[:,1], data[:,2] , c='yellow', marker='o', edgecolor='k', s=np.ones(len(data))*30, label='Sample particles')
ax.set_title('3D Plot Before Transformation')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.view_init(elev=45, azim=0)
ax.legend()
ax = fig.add_subplot(122, projection='3d')
sub_sample = transformed_hits_data.loc[list(truth_data[truth_data['particle_id']==sample_particle_id].index)]
data = np.array(sub_sample[['x2', 'y2', 'z2']])
ax.scatter(data[:,0], data[:,1], data[:,2] , c='red', marker='o', edgecolor='k', s=np.ones(len(data))*50, label='Same track particles')
sub_sample = transformed_hits_data.sample(100)
data = np.array(sub_sample[['x2', 'y2', 'z2']])
ax.scatter(data[:,0], data[:,1], data[:,2] , c='yellow', marker='o', edgecolor='k', s=np.ones(len(data))*30, label='Sample particles')
ax.set_title('3D Plot After Transformation')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.view_init(elev=45, azim=0)
ax.legend()
fig.tight_layout()
plt.show()

transformed_hits_data.head()

link_table = transformed_hits_data.join(truth_df[['particle_id']])[['x2', 'y2', 'z2', 'particle_id']]

link_table.head()

print(len(link_table[link_table['particle_id']==0]))
print(len(link_table[link_table['particle_id']!=0]))

sub_sample = link_table[link_table['particle_id']!=0]
fig = plt.figure(figsize=(15, 15))
elev_range = [0, 10, 90]
azim_range = [0, 10, 90]
for i, elev_angle in enumerate(elev_range):
    for j, azim_angle in enumerate(azim_range):
        ax = fig.add_subplot(331 + i*3 + j, projection='3d')
        for sample_particle_id in np.unique(sub_sample['particle_id'].values)[:100]:
            data = np.array(sub_sample[sub_sample['particle_id']==sample_particle_id][['x2', 'y2', 'z2']])
            ax.scatter(data[:,0], data[:,1], data[:,2], marker='o', edgecolor='black', s=np.ones(len(data))*30, alpha=0.5)
        ax.set_title('3D Plot Before Transformation')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.view_init(elev=elev_angle, azim=azim_angle)
        ax.legend()

fig.tight_layout()
plt.show()

sub_sample = link_table[link_table['particle_id']!=0]
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
for sample_particle_id in np.unique(sub_sample['particle_id'].values)[:100]:
    data = np.array(sub_sample[sub_sample['particle_id']==sample_particle_id][['x2', 'y2', 'z2']])
    ax.plot(data[:,0], data[:,1], data[:,2], '-', alpha=0.5, lw=4)
    ax.scatter(data[:,0], data[:,1], data[:,2], marker='o', edgecolor='black', s=np.ones(len(data))*30, alpha=0.5)    
ax.set_title('3D Plot Before Transformation')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.view_init(elev=90, azim=0)
ax.legend()
fig.tight_layout()
plt.show()

sub_sample = link_table[link_table['particle_id']!=0]
for sample_particle_id in np.unique(sub_sample['particle_id'].values)[:100]:
    data = np.array(sub_sample[sub_sample['particle_id']==sample_particle_id][['x2', 'y2', 'z2']])

plt.figure(figsize=(10,5))
plt.title('Histograms of number of particles')
plt.hist(sub_sample.groupby(['particle_id'])['x2'].count(), edgecolor='k', bins=20)
plt.xlabel('Number of connected components')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10,5))
plt.title('Histograms of z amplitude')
plt.hist((sub_sample.groupby(['particle_id']).max() - sub_sample.groupby(['particle_id']).min())['z2'].values, edgecolor='k', bins=20, log=True)
plt.xlabel('z axis amplitude of connected components')
plt.show()

sub_sample['r2'] = np.sqrt(sub_sample['x2']**2+sub_sample['y2']**2)

plt.figure(figsize=(10,5))
plt.title('Radius (Projected on (O, x, y)) amplitude for connected components')
plt.hist(sub_sample.groupby(['particle_id'])['r2'].max() - sub_sample.groupby(['particle_id'])['r2'].min(), bins=20, edgecolor='k', log=True)
plt.xlabel('Radius')
plt.show()

def loss(x, lambd=0):
    if x > 0:
        return max(0, x-lambd)
    else:
        return max(0, -(lambd+x))
x = np.linspace(-10, 10, 100)
y = [loss(x_, 2) for x_ in x]
plt.title('Loss function')
plt.axhline(0, color='black', alpha=0.5)
plt.axvline(0, color='black', alpha=0.5)
plt.xlabel('x')
plt.ylabel('loss')
plt.grid()
plt.plot(x, y, label='loss')
plt.legend()
plt.show()

def get_angle(x,y):
    if x > 0:
        return atan(y/x)
    else:
        return pi - atan(y/x)

sub_sample['r'] = np.sqrt(sub_sample['x2']**2+sub_sample['y2']**2)
sub_sample['theta'] = sub_sample.apply(lambda row: get_angle(row['x2'], row['y2']) - pi/2, axis=1)  # ?

def distance(point1, point2):
    return  loss(abs(float(point1['theta'])-float(point2['theta']))%360,50)+\
            loss(abs(float(point1['r'])-float(point2['r'])),10)+\
            loss(abs(float(point1['z2'])-float(point2['z2'])),10)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
for sample_particle_id in np.unique(sub_sample['particle_id'].values)[:100]:
    data = np.array(sub_sample[sub_sample['particle_id']==sample_particle_id][['r', 'theta']])
    ax.scatter(data[:,0]*np.cos(data[:,1]), data[:,0]*np.sin(data[:,1]), marker='o', edgecolor='black', s=np.ones(len(data))*30, alpha=0.5)
ax.set_title('3D Plot Before Transformation')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.legend()
fig.tight_layout()
plt.show()

table_ = sub_sample[sub_sample['theta'] < pi/2][sub_sample['theta'] > 0] # Centered on pi/4

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
for sample_particle_id in np.unique(table_['particle_id'].values)[:100]:
    data = np.array(table_[table_['particle_id']==sample_particle_id][['r', 'theta']])
    ax.plot(data[:,0]**2*np.cos(data[:,1]), data[:,0]**2*np.sin(data[:,1]), '-', alpha=0.5, lw=4)
    ax.scatter(data[:,0]**2*np.cos(data[:,1]), data[:,0]**2*np.sin(data[:,1]), marker='o', edgecolor='black', s=np.ones(len(data))*30, alpha=0.5)
ax.set_title('3D Plot Before Transformation')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.legend()

fig.tight_layout()
plt.show()

center = pi/4
print(center)
amplitude = pi/4
x = np.linspace(center - amplitude, center + amplitude, 100)
y = x-np.tanh((x-center)*2)*center
plt.plot(x, y)
plt.axhline(0, color='black')
plt.axvline(center - amplitude, color='black')
plt.axvline(center + amplitude, color='black')
plt.axhline(center, color='black')
plt.grid()

table_ = sub_sample[sub_sample['theta'] < pi/2][sub_sample['theta'] > 0]

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(131)
for sample_particle_id in np.unique(table_['particle_id'].values)[:100]:
    data = np.array(table_[table_['particle_id']==sample_particle_id][['r', 'theta']])
    ax.plot(data[:,0]**2*np.cos(data[:,1]), data[:,0]**2*np.sin(data[:,1]), '-', alpha=0.5, lw=2)
    ax.scatter(data[:,0]**2*np.cos(data[:,1]), data[:,0]**2*np.sin(data[:,1]), marker='o', edgecolor='black', s=np.ones(len(data))*30, alpha=0.5)
ax.set_title('3D Plot Before Transformation')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
table_['theta'] = table_['theta']*(1-table_['r']/table_['r'].max()) + (table_['theta'] - (np.tanh((table_['theta']-center)*2)*center).values)*(table_['r']/table_['r'].max())
ax = fig.add_subplot(132)
for sample_particle_id in np.unique(table_['particle_id'].values)[:100]:
    data = np.array(table_[table_['particle_id']==sample_particle_id][['r', 'theta']])
    x = data[:,0]**2*np.cos(data[:,1])
    y = data[:,0]**2*np.sin(data[:,1])
    ax.plot(x, y, '-', alpha=0.5, lw=2)
    ax.scatter(x, y, marker='o', edgecolor='black', s=np.ones(len(data))*30, alpha=0.5)
ax = fig.add_subplot(133)
for sample_particle_id in np.unique(table_['particle_id'].values)[:100]:
    data = np.array(table_[table_['particle_id']==sample_particle_id][['r', 'theta']])
    x = data[:,0]**2*np.cos(data[:,1])
    y = data[:,0]**2*np.sin(data[:,1])
    fish_eye = [0, 0]
    ax.plot(np.log(abs(x-fish_eye[0])), np.log(abs(y-fish_eye[1])), '-', alpha=0.5, lw=2)
    ax.scatter(np.log(abs(x-fish_eye[0])), np.log(abs(y-fish_eye[1])), marker='o', edgecolor='black', s=np.ones(len(data))*30, alpha=0.5)    
    
ax.set_title('3D Plot Before Transformation')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.legend()

fig.tight_layout()
plt.show()

eps_range = np.linspace(0.01, 0.1, 5)
min_samples_range = np.arange(1, 5)
for eps_ in eps_range:
    for min_samples_ in min_samples_range:

        dbs = DBSCAN(eps=eps_, min_samples=min_samples_)

        X = table_.drop('particle_id', axis=1)
        y = table_['particle_id']
        y_preds = dbs.fit_predict(X)
    
        print('eps:{}, min_samples:{}, score:{}'.format(eps_, min_samples_, metrics.adjusted_mutual_info_score(y, y_preds)))