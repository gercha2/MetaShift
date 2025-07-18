"""
Generate MetaDataset with train/test split 

"""

#CUSTOM_SPLIT_DATASET_FOLDER = '/data/MetaShift/Domain-Generalization-Cat-Dog'
CUSTOM_SPLIT_DATASET_FOLDER = '../Domain-Generalization-Bus-Truck'

import pandas as pd 
import seaborn as sns

import pickle
import numpy as np
import json, re, math
from collections import Counter, defaultdict
from itertools import repeat
import pprint
import os, errno
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shutil # for copy files
import networkx as nx # graph vis
import pandas as pd
from sklearn.decomposition import TruncatedSVD

import Constants
IMAGE_DATA_FOLDER          = Constants.IMAGE_DATA_FOLDER

from generate_full_MetaShift import preprocess_groups, build_subset_graph, copy_image_for_subject


def print_communities(subject_data, node_name_to_img_id, trainsg_dupes, subject_str):
    ##################################
    # Community detection 
    ##################################
    G = build_subset_graph(subject_data, node_name_to_img_id, trainsg_dupes, subject_str)

    import networkx.algorithms.community as nxcom

    # Find the communities
    communities = sorted(nxcom.greedy_modularity_communities(G), key=len, reverse=True)
    # Count the communities
    print(f"The graph has {len(communities)} communities.")
    for community in communities:
        community_merged = set()
        for node_str in community:
            node_str = node_str.replace('\n', '')
            node_image_IDs = node_name_to_img_id[node_str]
            community_merged.update(node_image_IDs)
            # print(node_str , len(node_image_IDs), end=';')

        print('total size:',len(community_merged))
        community_set = set([ x.replace('\n', '') for x in community])
        print(community_set, '\n\n')
    return G 



def parse_dataset_scheme(dataset_scheme, node_name_to_img_id, exclude_img_id=set(), split='test', copy=True):
    """
    exclude_img_id contains both trainsg_dupes and test images that we do not want to leak 
    """
    community_name_to_img_id = defaultdict(set)
    all_img_id = set()

    ##################################
    # Iterate subject_str: e.g., cat
    ##################################
    for subject_str in dataset_scheme:        
        ##################################
        # Iterate community_name: e.g., cat(sofa)
        ##################################
        for community_name in dataset_scheme[subject_str]:
            ##################################
            # Iterate node_name: e.g., 'cat(cup)', 'cat(sofa)', 'cat(chair)'
            ##################################
            for node_name in dataset_scheme[subject_str][community_name]:
                community_name_to_img_id[community_name].update(node_name_to_img_id[node_name] - exclude_img_id)
                all_img_id.update(node_name_to_img_id[node_name] - exclude_img_id)
            if copy:
                print(community_name, 'Size:', len(community_name_to_img_id[community_name]) )


        ##################################
        # Iterate community_name: e.g., cat(sofa)
        ##################################
        if copy:
            root_folder = os.path.join(CUSTOM_SPLIT_DATASET_FOLDER, split)
            copy_image_for_subject(root_folder, subject_str, dataset_scheme[subject_str], community_name_to_img_id, trainsg_dupes=set(), use_symlink=False) # use False to share 

    return community_name_to_img_id, all_img_id


def get_all_nodes_in_dataset(dataset_scheme):
    all_nodes = set()
    ##################################
    # Iterate subject_str: e.g., cat
    ##################################
    for subject_str in dataset_scheme:        
        ##################################
        # Iterate community_name: e.g., cat(sofa)
        ##################################
        for community_name in dataset_scheme[subject_str]:
            ##################################
            # Iterate node_name: e.g., 'cat(cup)', 'cat(sofa)', 'cat(chair)'
            ##################################
            for node_name in dataset_scheme[subject_str][community_name]:
                all_nodes.add(node_name)
    return all_nodes

def generate_splitted_metadaset():

    if os.path.isdir(CUSTOM_SPLIT_DATASET_FOLDER): 
        shutil.rmtree(CUSTOM_SPLIT_DATASET_FOLDER) 
    os.makedirs(CUSTOM_SPLIT_DATASET_FOLDER, exist_ok = False)


    node_name_to_img_id, most_common_list, subjects_to_all_set, subject_group_summary_dict = preprocess_groups(output_files_flag=False)

    ##################################
    # Removing ambiguous images that have both cats and dogs 
    ##################################
    trainsg_dupes = node_name_to_img_id['bus(truck)'] # can also use 'truck(bus)'
    subject_str_to_Graphs = dict()


    for subject_str in ['bus', 'truck']:
        subject_data = [ x for x in subject_group_summary_dict[subject_str].keys() if x not in ['bus(truck)', 'truck(bus)'] ]
        # print('subject_data', subject_data)
        ##################################
        # Print detected communities in Meta-Graph
        ##################################
        G = print_communities(subject_data, node_name_to_img_id, trainsg_dupes, subject_str) # print detected communities, which guides us the train/test split. 
        subject_str_to_Graphs[subject_str] = G




    train_set_scheme = {
        # Note: these comes from copy-pasting the community detection results of bus & truck. 
        'bus': {
            # The bus training data is always bus(\emph{clock + traffic light}) 
            'bus(clock)': {'bus(clock)', 'bus(woman)', 'bus(person)'}, #A
            'bus(traffic light)':  {'bus(pole)', 'bus(street light)', 'bus(suv)', 'bus(traffic light)', 'bus(fire hydrant)'}, #B
        }, 
        'truck': {
            # Experiment 1: the dog training data is dog(\emph{cone + fence}) communities, and its distance to truck(\emph{airplane}) is $d$=0.81
            'truck(cone)': {'truck(cone)', 'truck(cones)', 'truck(airplane)', 'truck(sign)', 'truck(ground)'}, #d = 0.39
            'truck(fence)': {'truck(fence)', 'truck(horse)', 'truck(grass)', 'truck(house)', 'truck(palm tree)', 'truck(trees)'}, #d = 0.39

            # Experiment 2: the dog training data is dog(\emph{bag + box}), and its distance to dog(\emph{shelf}) is $d$=1.20
            'truck(bike)': {'truck(bike)', 'truck(helmet)', 'truck(bicycle)', 'truck(motorbicycle)'}, #Exp2 d=1.34
            'truck(mirror)': {'truck(mirror)', 'truck(taxi)', 'truck(cars)', 'truck(van)', 'truck(car)'} , #Exp2 d=0.73 B          

            # Experiment 3: the dog training data is dog(\emph{bench + bike}) with distance $d$=1.42
            'truck(flag)': {'truck(flag)', 'truck(american flag)', 'flag(sign)'} , # Exp3 d=1.03
            'truck(tower)': {'truck(tower)', 'truck(ladder)', 'truck(building)'}, #Exp3 d=1.12 B

            # Experiment 4: the dog training data is dog(\emph{boat + surfboard}) with distance $d$=1.52
            'truck(traffic light)': {'truck(traffic light)', 'truck(vehicles)', 'truck(vehicle)', 'truck(car)', 'truck(suv)'}, #Exp4 d=1.53
            'truck(dog)': {'truck(dog)', 'truck(horse)'}, # 'dog(ball)', #Exp4 d=1.40 
        }
    }

    test_set_scheme = {
        'bus': {
            'bus(airplane)': {'bus(airplane)', 'bus(truck)', 'bus(bus driver)', 'bus(sky)'}, #Exp1 d=0.99
        },
        'truck': {
            # In MetaDataset paper, the test images are all dogs. However, for completeness, we also provide cat images here. 
            'truck(airplane)': {'truck(airplane)', 'truck(vehicle)', 'truck(cart)', 'truck(vehicles)'}, # Exp1 d=0.81
        },
    }

    additional_test_set_scheme = {
        'bus': {
            'bus(people)': {'bus(people)', 'bus(man)', 'bus(person)', 'bus(lady)', 'bus(child)', 'bus(pedestrian)', 'bus(woman)', 'bus(girl)', 'bus(men)'},
            'bus(cellphone)': {'bus(cellphone)', 'bus(camera)', 'bus(phone)'}, 
            'bus(house)': {'bus(house)', 'bus(bridge)', 'bus(wall)', 'bus(trees)'}, 
            'bus(vehicles)': {'bus(vehicles)', 'bus(taxi)', 'bus(vehicle)', 'bus(van)', 'bus(car)', 'bus(cars)', 'bus(suv)'}, 
            'bus(street light)': {'bus(street light)', 'bus(traffic light)', 'bus(fire hydrant)', 'bus(pole)', 'bus(lamp)'},
        },
        'truck': {
            'truck(people)': {'truck(people)', 'truck(woman)', 'truck(child)',  'truck(men)', 'truck(girl)', 'truck(child)'}, 
            'truck(motorcycle)': {'truck(motorcycle)', 'truck(bike)', 'truck(bicycle)'}, 
            'truck(table)': {'truck(table)', 'truck(chair)', 'truck(bed)'}, 
            'truck(vehicles)': {'truck(vehicle)', 'truck(vehicles)', 'truck(train)', 'truck(taxi)', 'truck(car)', 'truck(van)', 'truck(suv)'},
        },
    }


    print('========== test set info ==========')
    test_community_name_to_img_id, test_all_img_id = parse_dataset_scheme(test_set_scheme, node_name_to_img_id, exclude_img_id=trainsg_dupes, split='test')
    # print('test_all_img_id', len(test_all_img_id))
    print('========== train set info ==========')
    train_community_name_to_img_id, train_all_img_id = parse_dataset_scheme(train_set_scheme, node_name_to_img_id, exclude_img_id=test_all_img_id.union(trainsg_dupes), split='train')
    print('========== additional test set info ==========')
    additional_test_community_name_to_img_id, additional_test_all_img_id = parse_dataset_scheme(additional_test_set_scheme, node_name_to_img_id, exclude_img_id=train_all_img_id.union(trainsg_dupes), split='test')


    ##################################
    # **Quantifying the distance between train and test subsets**
    # Please be advised that before making MetaShift public, 
    # we have made further efforts to reduce the label errors propagated from Visual Genome. 
    # Therefore, we expect a slight change in the exact experiment numbers.  
    ##################################
    
    print('========== Quantifying the distance between train and test subsets ==========')
    test_community_name_to_img_id, _ = parse_dataset_scheme(test_set_scheme, node_name_to_img_id, exclude_img_id=trainsg_dupes, split='test', copy=False)
    train_community_name_to_img_id, _ = parse_dataset_scheme(train_set_scheme, node_name_to_img_id, exclude_img_id=trainsg_dupes, split='train', copy=False)
    additional_test_community_name_to_img_id, _ = parse_dataset_scheme(additional_test_set_scheme, node_name_to_img_id, exclude_img_id=trainsg_dupes, split='test')

    community_name_to_img_id = test_community_name_to_img_id.copy()
    community_name_to_img_id.update(train_community_name_to_img_id)
    community_name_to_img_id.update(additional_test_community_name_to_img_id)
    dog_community_name_list = sorted(train_set_scheme['truck']) + sorted(test_set_scheme['truck']) + sorted(additional_test_set_scheme['truck'])
    
    G = build_subset_graph(dog_community_name_list, community_name_to_img_id, trainsg_dupes=set(), subject_str=None)

    spectral_pos = nx.spectral_layout(
        G=G, 
        dim=5,
        )
    
    for subset_A, subset_B in [
        ['truck(cone)', 'truck(fence)'],
        ['truck(bike)', 'truck(mirror)'],
        ['truck(flag)', 'truck(tower)'],
        ['truck(traffic light)', 'truck(dog)'],
    ]:
        distance_A = np.linalg.norm(spectral_pos[subset_A.replace('(', '\n(')] - spectral_pos['truck\n(airplane)'])
        distance_B = np.linalg.norm(spectral_pos[subset_B.replace('(', '\n(')] - spectral_pos['truck\n(airplane)'])
        
        print('Distance from {}+{} to {}: {}'.format(
            subset_A, subset_B, 'truck(airplane)', 
            0.5 * (distance_A + distance_B)
            )
        )

        
    return

if __name__ == '__main__':
    generate_splitted_metadaset()

