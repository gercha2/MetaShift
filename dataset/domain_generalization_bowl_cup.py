"""
Generate MetaDataset with train/test split 

"""

#CUSTOM_SPLIT_DATASET_FOLDER = '/data/MetaShift/Domain-Generalization-Cat-Dog'
CUSTOM_SPLIT_DATASET_FOLDER = '../Domain-Generalization-Bowl-Cup'

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
    trainsg_dupes = node_name_to_img_id['bowl(cup)'] # can also use 'cup(bowl)'
    subject_str_to_Graphs = dict()


    for subject_str in ['bowl', 'cup']:
        subject_data = [ x for x in subject_group_summary_dict[subject_str].keys() if x not in ['bowl(cup)', 'cup(bowl)'] ]
        # print('subject_data', subject_data)
        ##################################
        # Print detected communities in Meta-Graph
        ##################################
        G = print_communities(subject_data, node_name_to_img_id, trainsg_dupes, subject_str) # print detected communities, which guides us the train/test split. 
        subject_str_to_Graphs[subject_str] = G


    train_set_scheme = {
        # Note: these comes from copy-pasting the community detection results of cat & dog. 
        'bowl': {
            # The bowl training data is always bowl(\emph{sofa + bed}) 
            #'bowl(sofa)': {'bowl(cup)', 'bowl(sofa)', 'bowl(chair)'},
            'bowl(fruit)': {'bowl(fruit)', 'bowl(apple)', 'bowl(apples)', 'bowl(banana)'}, #A
            'bowl(tray)':  {'bowl(tray)', 'bowl(donut)', 'bowl(spoon)', 'bowl(chopsticks)', 'bowl(juice)', 'bowl(napkin)'}, #B
        }, 
        'cup': {
            # Experiment 1: the cup training data is cup(\emph{cabinet + bed}) communities, and its distance to cup(\emph{shelf}) is $d$=0.44. 
            'cup(knife)': {'cup(knife)', 'cup(placemat)', 'cup(food)', 'cup(fork)', 'cup(plate)' }, #A
            'cup(tray)': {'cup(tray)', 'cup(placemat)', 'cup(plate)', 'cup(bowl)', 'cup(napkin)', 'cup(knife)', 'cup(spoon)', 'cup(fork)'}, #B

            # Experiment 2: the dog training data is dog(\emph{bag + box}), and its distance to dog(\emph{shelf}) is $d$=0.71. 
            'cup(water)': {'cup(water)', 'cup(salt shaker)', 'cup(soup)', 'cup(food)', 'cup(plate)', 'cup(bowl)'}, #Exp2 d=1.34
            'cup(cabinet)': {'cup(cabinet)', 'cup(counter)', 'cup(oven)', 'cup(stove)', 'cup(utensils)', 'cup(refrigerator)', 'cup(microwave)', 'cup(toilet)', 'cup(faucet)', 'cup(towel)', 'cup(sink)', 'cup(countertop)', 'cup(drawer)'} , #Exp2 d=0.73 B          

            # Experiment 3: the cup training data is cup(\emph{bench + bike}) with distance $d$=1.12
            'cup(computer)': {'cup(computer)', 'cup(screen)', 'cup(phone)', 'cup(laptop)', 'cup(speaker)', 'cup(keyboard)', 'cup(desk)', 'cup(monitor)'} , # Exp3 d=1.03
            'cup(lamp)': {'cup(lamp)', 'cup(curtain)', 'cup(bed)', 'cup(chair)', 'cup(picture)', 'cup(couch)', 'cup(pillow)', 'cup(remote control)', 'cup(bed)', 'cup(blanket)'}, #Exp3 d=1.12 B

            # Experiment 4: the cup training data is cup(\emph{boat + surfboard}) with distance $d$=1.43.   
            'cup(toilet)': {'cup(toilet)', 'cup(mirror)', 'cup(towel)', 'cup(faucet)', 'cup(sink)', 'cup(counter)', 'cup(cabinet)'}, #Exp4 d=1.5
            'cup(box)': {'cup(box)', 'cup(bag)', 'cup(purse)', 'cup(backpack)'}, # 'dog(ball)', #Exp4 d=1.40 
        }
    }

    test_set_scheme = {
        'bowl': {
            'bowl(coffee)': {'bowl(coffee)', 'bowl(coffee maker)', 'bowl(milk)', 'bowl(counter)'},
        },
        'cup': {
            'cup(coffee)': {'cup(coffee)', 'cup(saucer)', 'cup(toast)', 'cup(spoon)', 'cup(plate)', 'cup(tea)', 'cup(tray)'}, 
        },
    }

    additional_test_set_scheme = {
        'bowl': {
            'bowl(toilet)': {'bowl(toilet)', 'bowl(toilet paper)', 'bowl(soap)', 'bowl(towel)', 'bowl(sink)', 'cup(mirror)'},
            'bowl(vegetables)': {'bowl(carrots)', 'bowl(broccoli)', 'bowl(carrot)', 'bowl(onions)', 'bowl(onion)', 'cup(tomatoes)', 'cup(cucumber)', 'cup(lettuce)', 'cup(pepper)', 'cup(potato)', 'cup(potatoes)', 'cup(tomato)'}, 
            'bowl(bed)': {'bowl(bed)', 'bowl(blanket)', 'bowl(pillow)'}, 
            'bowl(blender)': {'bowl(blender)', 'bowl(stove)', 'bowl(oven)', 'bowl(refrigerator)', 'bowl(dishwasher)', 'bowl(cabinets)', 'bowl(shelf)'}, 
            'bowl(pizza)': {'bowl(pizza)', 'bowl(sandwich)', 'bowl(fries)', 'bowl(salad)', 'bowl(bacon)', 'cup(eggs)', 'cup(cake)'},
        },
        'cup': {
            'cup(people)': {'cup(people)', 'cup(man)', 'cup(woman)',  'cup(person)', 'cup(baby)', 'cup(girl)', 'cup(lady)', 'cup(child)'}, 
            'cup(table)': {'cup(table)', 'cup(wine glass)', 'cup(wine)', 'cup(coffee cup)', 'cup(beer)','cup(drink)', 'cup(water bottle)', 'cup(can)'}, 
            'cup(beverage)': {'cup(beverage)', 'cup(milk)', 'cup(tea)', 'cup(juice)', 'cup(water bottle)', 'cup(soda)', 'cup(can)', 'cup(water)', 'cup(coffee)', 'cup(juice)'}, 
            'cup(food)': {'cup(food)', 'cup(sandwich)', 'cup(eggs)', 'cup(hot dog)', 'cup(bread)', 'cup(salad)', 'cup(meat)', 'cup(egg)'},
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
    dog_community_name_list = sorted(train_set_scheme['cup']) + sorted(test_set_scheme['cup']) + sorted(additional_test_set_scheme['cup'])
    
    G = build_subset_graph(dog_community_name_list, community_name_to_img_id, trainsg_dupes=set(), subject_str=None)

    spectral_pos = nx.spectral_layout(
        G=G, 
        dim=5,
        )
    
    for subset_A, subset_B in [
        ['cup(knife)', 'cup(tray)'],
        ['cup(water)', 'cup(cabinet)'],
        ['cup(computer)', 'cup(lamp)'],
        ['cup(toilet)', 'cup(box)'],
    ]:
        distance_A = np.linalg.norm(spectral_pos[subset_A.replace('(', '\n(')] - spectral_pos['cup\n(coffee)'])
        distance_B = np.linalg.norm(spectral_pos[subset_B.replace('(', '\n(')] - spectral_pos['cup\n(coffee)'])
        
        print('Distance from {}+{} to {}: {}'.format(
            subset_A, subset_B, 'cup(coffee)', 
            0.5 * (distance_A + distance_B)
            )
        )

        
    return

if __name__ == '__main__':
    generate_splitted_metadaset()

