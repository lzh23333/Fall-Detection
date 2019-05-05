from feature_extract import *
import numpy as np
import pandas as pd
import os
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import xgboost as xgb

def init_files(dataset,actor_range,file_list):
    tmp_files = {}
    for act in act1+act2:
        tmp_files[act] = []
    filenames = {}
    for act in tmp_files.keys():
        print(act)
        filenames[act] = get_data_list(dataset,act,actor_range,file_list)
        for f in filenames[act]:
            tmp = read_skeleton(pd.read_csv(f,engine='python'))
            tmp = shape_data(tmp)
            tmp_files[act].append(tmp)
        print([len(x) for key,x in tmp_files.items()])
    return tmp_files

def main():

    dataset = 'E:/Course/媒体与认知/fall_detection/database'
    act1 = ['grasp','lay','sit','walk']
    act2 = ['back', 'EndUpSit', 'front', 'side']
    acttype = ['ADL','Fall']
    actor_range = list(range(1,10))
    train_list = [1,2]
    test_list = [3]

    train_files = init_files(dataset,actor_range,train_list)
    test_files = init_files(dataset,actor_range,test_list)

    frame_num = 10
    step = 5
    train_data =[]
    test_data = []
    for act in train_files.keys():
        label = 0
        if act in act2:
            label = 1
        for frames in train_files[act]:
            tmp = feature(frames,frame_num,step,label)
            train_data.append(tmp)
        for frames in test_files[act]:
            test_data.append(feature(frames,frame_num,step,label))
    
    train = None
    test = None
    for array in train_data:
        if train is None:
            train = array
        else:
            train = np.vstack((train,array))
    for array in test_data:
        if test is None:
            test = array
        else:
            test = np.vstack((test,array))
    
    print('train feature size:' + str(train.shape))
    print('test feature size:' + str(test.shape))

    '''
    clf = LinearSVC(loss='hinge')
    print(clf)
    print('begin trainning')
    clf.fit(train[:,:-2],train[:,-1])
    print(clf)
    pred = clf.predict(test[:,:-2])
    '''
    dtrain = xgb.DMatrix(train[:,:-2],label = train[:,-1])
    dtest = xgb.DMatrix(test[:,:-2],label = test[:,-1])
    num_round = 20
    param = {'max_depth':3, 'eta':1, 'silent':0, 'objective':'binary:logistic' }
    bst = xgb.train(param,dtrain,num_round)
    pro = 0.9
    pred = bst.predict(dtest)>pro


    print(classification_report(y_true=test[:,-1],y_pred=pred))
    plt.plot(pred,'r')
    plt.plot(test[:,-1],'b')
    plt.show()

if __name__ == '__main__':
    main()