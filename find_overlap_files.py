import os
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Manager
mode='val'
dataset = 'nusc'
if dataset=='nusc':
    with open('/data/laiyan/airs/VFDepth/dataset/{}/info_{}.pkl'.format(dataset, mode), 'rb') as f:
        info = pickle.load(f)
    cams=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
    min_depth=0.1
    max_depth=80
if dataset=='ddad':
    with open('/data/laiyan/ssd/ddad/meta_data/info_{}.pkl'.format(mode), 'rb') as f:
        info = pickle.load(f)

    ddad_depth_path = '/data/laiyan/ssd/ddad/depth'
    cams=['CAMERA_01', 'CAMERA_05', 'CAMERA_06', 'CAMERA_07', 'CAMERA_08', 'CAMERA_09']
    min_depth = 0.1
    max_depth = 200


with open("/data/laiyan/airs/SurroundDepth/datasets/{}/{}.txt".format(dataset,mode),'r') as f:
    sr_val = f.readlines()


# valid_names=[]
# ss_list=[]
#
# for name in tqdm(sr_val[:100]):
#     name = name.strip()
#     data = info[name]
#     flag=True
#     for cam in cams:
#         if dataset=='nusc':
#             rgb_name = '/data/laiyan/datasets/nuscenes/v1.0/'+data[cam]['rgb_filenames'][0]
#             overlap_depth = np.load(rgb_name.replace('jpg', 'npz').replace('samples', 'overlap_depth/samples'))['arr_0']
#         if dataset=='ddad':
#             scene_name = info[name]['scene_name']
#             overlap_depth = np.load(os.path.join(ddad_depth_path, scene_name, 'depth_overlap',cam, name + '.npz'))['arr_0']
#         ss = np.count_nonzero(overlap_depth)
#         ss_list.append(ss)
#         if ss<=50:
#             print('empty')
#             flag=False
#     if flag:
#         valid_names.append(name)
# plt.hist(ss_list,bins=200);plt.show()
# print(len(valid_names))

def generate(index):
    print(index)
    name = sr_val[index]
    name = name.strip()
    data = info[name]
    flag = True
    for cam in cams:
        if dataset == 'nusc':
            rgb_name = '/data/laiyan/datasets/nuscenes/v1.0/' + data[cam]['rgb_filenames'][0]
            overlap_depth = np.load(rgb_name.replace('jpg', 'npz').replace('samples', 'overlap_depth/samples'))[
                'arr_0']
        if dataset == 'ddad':
            scene_name = info[name]['scene_name']
            overlap_depth = np.load(os.path.join(ddad_depth_path, scene_name, 'depth_overlap', cam, name + '.npz'))[
                'arr_0']
        ss = np.count_nonzero(overlap_depth)
        ss_list.append(ss)
        if ss <= 10:
            print('empty')
            flag = False


    if flag:
        valid_names.append(name)

with Manager() as manager:
    valid_names=manager.list()
    ss_list=manager.list()



    p = mp.Pool(20)
    info_list = [i for i in range(len(sr_val[:]))]
    p.map_async(generate, info_list)
    p.close()
    p.join()




    print('--------------------')
    plt.hist(ss_list,bins=200);plt.show()
    print(len(valid_names))
    with open("/data/laiyan/airs/SurroundDepth/datasets/{}/{}_overlap.txt".format(dataset,mode),'w') as f:
        f.writelines([s + '\n' for s in valid_names])
