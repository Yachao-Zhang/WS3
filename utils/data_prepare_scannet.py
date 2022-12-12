from sklearn.neighbors import KDTree
from os.path import join, exists, dirname, abspath
import numpy as np
import json
import pandas as pd
import os, sys, glob, pickle
from os import makedirs, listdir
BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import write_ply
from helper_tool import DataProcessing as DP
from utils.ply import read_ply, write_ply

dataset_path = '../scannet/'
sub_grid_size = 0.04
original_pc_folder = join(dirname(dataset_path), 'original_ply')
sub_pc_folder = join(dirname(dataset_path), 'input_{:.3f}'.format(sub_grid_size))
os.mkdir(original_pc_folder) if not exists(original_pc_folder) else None
os.mkdir(sub_pc_folder) if not exists(sub_pc_folder) else None
out_format = '.ply'
label_values=[0,1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]

def convert_pc2ply(xyz,colors,labels, save_path):

    write_ply(join(original_pc_folder, save_path.split('/')[-1] + '.ply'), (xyz, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
    # save sub_cloud and KDTree file
    sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, colors, labels, sub_grid_size)
    sub_colors = sub_colors / 255.0
    sub_ply_file = join(sub_pc_folder, save_path.split('/')[-1] + '.ply')
    write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

    search_tree = KDTree(sub_xyz)
    kd_tree_file = join(sub_pc_folder, str(save_path.split('/')[-1]) + '_KDTree.pkl')
    with open(kd_tree_file, 'wb') as f:
        pickle.dump(search_tree, f)

    proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
    proj_idx = proj_idx.astype(np.int32)
    proj_save = join(sub_pc_folder, str(save_path.split('/')[-1]) + '_proj.pkl')
    with open(proj_save, 'wb') as f:
        pickle.dump([proj_idx, labels], f)

def load_ply_labels(scans, train):
    scans_path = dataset_path + scans
    scenes = np.sort([f for f in listdir(scans_path)])
    N = len(scenes)
    for i, scene in enumerate(scenes):
        print(scene)
        # Check if file already done
        if exists( scene + '.ply'):
            continue
        ply_patch = join(scans_path, scene, scene + '_vh_clean_2.ply')
        vertex_data, faces = read_ply(ply_patch, triangular_mesh=True)
        vertices = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
        vertices_colors = np.vstack((vertex_data['red'], vertex_data['green'], vertex_data['blue'])).T

        vertices_labels = np.zeros(vertices.shape[0], dtype=np.int32)
        if train:
            with open(join(scans_path, scene, scene + '_vh_clean_2.0.010000.segs.json'), 'r') as f:
                segmentations = json.load(f)
            segIndices = np.array(segmentations['segIndices'])
            with open(join(scans_path, scene, scene + '_vh_clean.aggregation.json'), 'r') as f:
                aggregation = json.load(f)
            # Loop on object to classify points
            for segGroup in aggregation['segGroups']:
                c_name = segGroup['label']
                if c_name in names1:
                    nyuID = annot_to_nyuID[c_name]
                    if nyuID in label_values:
                        for segment in segGroup['segments']:
                            vertices_labels[segIndices == segment] = nyuID
        convert_pc2ply(vertices, vertices_colors, vertices_labels, scene)

if __name__ == '__main__':
    label_files = join(dataset_path, 'scannetv2-labels.combined.tsv')
    with open(label_files, 'r') as f:
        lines = f.readlines()
        names1 = [line.split('\t')[1] for line in lines[1:]]
        IDs = [int(line.split('\t')[4]) for line in lines[1:]]
        annot_to_nyuID = {n: id for n, id in zip(names1, IDs)}
    load_ply_labels('scans', True)
    load_ply_labels('scans_test', False)



    #

