import numpy as np
import pandas as pd
import os

act1 = ['grasp','lay','sit','walk']
act2 = ['back', 'EndUpSit', 'front', 'side']
acttype = ['ADL','Fall']
speed_boudary = 0.05
SpineBase = 0
Head = 3
ShoulderLeft = 4
ElbowLeft = 5
HandLeft = 7
ShoulderRight = 8
ElbowRight = 9
HandRight = 11
HipLeft = 12
KneeLeft = 13
FootLeft = 15
HipRight = 16
KneeRight = 17
FootRight = 19
SpineShoulder = 20
joint_list = [
    SpineBase,Head,SpineShoulder,ShoulderLeft,ShoulderRight,
    ElbowLeft,ElbowRight,HandLeft,HandRight,HipLeft,HipRight,
    KneeLeft,KneeRight,FootLeft,FootRight
]
angles_to_calculate = [
    (HandLeft,ElbowLeft,ShoulderLeft),
    (HandRight,ElbowRight,ShoulderRight),
    (HipLeft,KneeLeft,FootLeft),
    (HipRight,KneeRight,FootRight),
]

def get_data_list(dataset_path,act,actor_range,file_list):
    # 返回一个包含所有数据文件名的列表
    # actor_range: 
    # file_num: 每个actor选取文件数
    # act: 动作类型
    actor_List = [os.path.join(dataset_path,'Data'+str(i)) for i in actor_range]
    file_names = []
    acttype = 'Fall'
    if act in act1:
        acttype = 'ADL'
    for actor_floder in actor_List:
        act_path = os.path.join(actor_floder,acttype,act)
        for x in file_list:
            file_names.append(os.path.join(act_path,str(x),'Body','Fileskeleton.csv'))
    return file_names

def read_skeleton(df):
    #df is a pandas dataframe
    num_row = df.shape[0]
    skeleton_data = []
    idx = 0
    while idx < num_row:
        if df.iloc[idx,3] == 2:
            skeleton_data.append(df.iloc[idx:idx+25,:3].values)
            idx += 25
        else:
            idx += 1
    return skeleton_data

def extract_action_frames(speed,boundary):
    max_speed = np.max(speed)
    speed = speed/max_speed
    start = 0
    end = len(speed)
    for idx in range(end):
        if speed[idx] >= boundary:
            start = idx
            break
    for idx in range(end-1,start,-1):
        if speed[idx] >= boundary:
            end = idx
            break
    return (start,end)

def shape_data(data_list):
    # data_list:为连续的帧的列表
    # 每帧代表骨骼点
    centers = []
    for idx in range(len(data_list)):
        centers.append(np.mean(data_list[idx],axis=0))
        #centers.append(data_list[idx][1,:])
    speed = np.diff(centers,axis=0)**2
    speed = np.sqrt(np.sum(speed,axis=1))
    (start,end) = extract_action_frames(speed,speed_boudary)
    return data_list[start:end+1]

def feature(skeleton_frames,frame_num,step,label):
    # skeleton_frames: 包含骨骼点的list
    # 返回一个array
    length = len(skeleton_frames)
    feature = []
    for idx in range(0,length-frame_num,step):
        tmp_frames = [skeleton_frames[idx+i] for i in range(frame_num)]

        # get speed feature 
        centers = [np.mean(x,axis=0) for x in tmp_frames]
        #centers = [x[1,:] for x in tmp_frames]
        centers = np.array(centers)
        speed = np.diff(centers,axis=0)**2
        speed = np.sqrt(np.sum(speed,axis=1))

        # get skeleton 
        tmp_frames = np.array(tmp_frames).flatten()

        # get feature
        feature.append(np.hstack((tmp_frames,speed,np.array([label]))))
        #feature.append(np.hstack((tmp_frames,np.array([label]))))
    #for frame in feature:
        #print(frame.shape)
    #print(len(feature))
    #print(np.array(feature).shape)
    return np.array(feature)

def get_feature(skeleton_frames):
    length = len(skeleton_frames)
    feature = []
    
    tmp_frames = skeleton_frames
    # get speed feature 
    centers = [np.mean(x,axis=0) for x in tmp_frames]
    #centers = [x[1,:] for x in tmp_frames]
    centers = np.array(centers)
    speed = np.diff(centers,axis=0)**2
    speed = np.sqrt(np.sum(speed,axis=1))

    # get skeleton 
    tmp_frames = np.array(tmp_frames).flatten()

    # get feature
    return np.hstack((tmp_frames,speed))
    

def feature2(skeleton_frames, frame_num, step, label):
    # skeleton_frames: 包含骨骼点的list
    # floor_clip_frames: 包含地面方程的list
    # 返回一个array

    length = len(skeleton_frames)
    feature = []
    for idx in range(0,length-frame_num,step):
        tmp_frames = [skeleton_frames[idx+i] for i in range(frame_num)]
        angles = [calc_skeleton_angles(i) for i in tmp_frames]
        angles = np.array(angles).flatten()
        select_frames = [i[joint_list,:] for i in tmp_frames]
        select_frames = np.array(select_frames).flatten()

        feature.append(np.hstack((select_frames,angles,np.array(label))))

    return np.array(feature)




def calc_cos(vec1,vec2):
    x = np.linalg.norm(vec1)* np.linalg.norm(vec2)
    return vec1.dot(vec2)/(x)

def calc_skeleton_angles(skeleton):
    angles = []
    for x in angles_to_calculate:
        vec1 = skeleton[x[0],:]-skeleton[x[1],:]
        vec2 = skeleton[x[2],:]-skeleton[x[1],:]
        angles.append(calc_cos(vec1,vec2))
    return np.array(angles)