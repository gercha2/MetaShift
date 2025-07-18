"""
Generate MetaDataset with train/test split 

"""

#CUSTOM_SPLIT_DATASET_FOLDER = '/data/MetaShift/Domain-Generalization-Cat-Dog'
CUSTOM_SPLIT_DATASET_FOLDER = '../Domain-Generalization-elephant-horse'

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
    trainsg_dupes = node_name_to_img_id['horse(elelphant)'] # can also use 'dog(cat)'
    subject_str_to_Graphs = dict()


    for subject_str in ['elephant', 'horse']:
        subject_data = [ x for x in subject_group_summary_dict[subject_str].keys() if x not in ['elephant(horse)', 'horse(elephant)'] ]
        # print('subject_data', subject_data)
        ##################################
        # Print detected communities in Meta-Graph
        ##################################
        G = print_communities(subject_data, node_name_to_img_id, trainsg_dupes, subject_str) # print detected communities, which guides us the train/test split. 
        subject_str_to_Graphs[subject_str] = G




    train_set_scheme = {
        # Note: these comes from copy-pasting the community detection results of cat & dog. 
        'elephant': {
            # The cat training data is always cat(\emph{sofa + bed}) 
            #'cat(sofa)': {'cat(cup)', 'cat(sofa)', 'cat(chair)'},
            'elephant(fence)': {'elephant(fence)', 'elephant(hay)', 'elephant(rock)', 'elephant(wall)', 'elephant(door)', 'elephant(ground)'}, #A
            'elephant(rock)':  {'elephant(rock)', 'elephant(rocks)', 'elephant(water)', 'elephant(fence)', 'elephant(wall)', 'elephant(baby)'}, #B
        }, 
        'horse': {
            # Experiment 1: the dog training data is dog(\emph{cabinet + bed}) communities, and its distance to dog(\emph{shelf}) is $d$=0.44. 
            'horse(dirt)': {'horse(dirt)', 'horse(cowboy)', 'horse(fence)', 'dog(chair)', 'dog(window)'}, #A
            'horse(trees)': {'horse(trees)', 'horse(mountains)', 'horse(barn)', 'horse(jockey)', 'horse(grass)', 'horse(house)', 'horse(bench)', 'horse(fence)'}, #B

            # Experiment 2: the dog training data is dog(\emph{bag + box}), and its distance to dog(\emph{shelf}) is $d$=0.71. 
            'horse(fence)': {'horse(fence)', 'horse(pole)', 'horse(grass)', 'horse(trees)'}, #Exp2 d=1.34
            'horse(helmet)': {'horse(helmet)', 'horse(flag)', 'horse(glasses)', 'horse(lady)', 'horse(blanket)'} , #Exp2 d=0.73 B
            
            # Experiment 3: the horse training data is horse(\emph{bench + bike}) with distance $d$=1.12
            'horse(car)': {'horse(car)', 'horse(tower)', 'horse(bus)', 'horse(hat)', 'horse(building)'} , # Exp3 d=1.03
            'horse(wagon)': {'horse(wagon)', 'horse(people)', 'horse(carriage)', 'horse(person)', 'horse(building)'}, #Exp3 d=1.12 B

            # Experiment 4: the horse training data is horse(\emph{boat + surfboard}) with distance $d$=1.43.   
            'horse(statue)': {'horse(statue)', 'horse(clock)', 'horse(people)', 'horse(building)'}, #Exp4 d=1.53
            'horse(cart)': {'horse(cart)', 'horse(wall)', 'horse(rope)', 'horse(tree)'}, # 'dog(ball)', #Exp4 d=1.40 
        }
    }

    test_set_scheme = {
        'horse': {
            #'cat(shelf)': {'cat(container)', 'cat(shelf)', 'cat(vase)', 'cat(bookshelf)', 'cat(floor)', 'cat(table)', 'cat(books)', 'cat(book)'},
            'horse(barn)': {'horse(barn)', 'horse(animal)', 'horse(house)', 'horse(fence)', 'horse(jockey)'},
        },
        'elephant': {
            # In MetaDataset paper, the test images are all dogs. However, for completeness, we also provide cat images here. 
            #'dog(shelf)': {'dog(desk)', 'dog(screen)', 'dog(laptop)', 'dog(shelf)', 'dog(picture)', 'dog(chair)'}, 
            'elephant(house)': {'elephant(house)', 'elephant(door)', 'elephant(building)'},
            },
    }

    additional_test_set_scheme = {
        'elephant': {
            'elephant(trees)': {'elephant(trees)', 'elephant(leaves)', 'elephant(tree)', 'elephant(branch)'},
            'elephant(animal)': {'elephant(bird)', 'elephant(animal)', }, 
            'elephant(people)': {'elephant(people)', 'elephant(boy)', 'elephant(girl)', 'elephant(person)', 'elephant(woman)', 'elephant(man)', 'elephant(child)'}, 
            'elephant(building)': {'elephant(building)', 'elephant(wall)', 'elephant(door)'}, 
            'elephant(bush)': {'elephant(rock)', 'elephant(bush)', 'elephant(grass)', 'elephant(tree)', 'elephant(trees)'},
        },
        'horse': {
            'horse(car)': {'horse(car)', 'horse(bus)', 'horse(cart)', 'horse(carriage)', 'horse(truck)'}, 
            'horse(building)': {'horse(building)', 'horse(statue)', 'horse(tower)'}, 
            'horse(people)': {'horse(people)', 'horse(woman)', 'horse(person)', 'horse(child)', 'horse(men)', 'horse(girl)', 'horse(boy)'}, 
            'horse(animal)': {'horse(animal)', 'horse(cow)', 'horse(dog)', 'horse(bird)'},
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
    #horse_community_name_list = sorted(train_set_scheme['horse']) + sorted(test_set_scheme['horse']) + sorted(additional_test_set_scheme['horse'])
    
    # Combine all horse community names from all schemes
    horse_community_name_set = set(train_set_scheme['horse']) | set(test_set_scheme['horse']) | set(additional_test_set_scheme['horse'])
    
    # Only keep those that are present in community_name_to_img_id
    horse_community_name_list = [k for k in sorted(horse_community_name_set) if k in community_name_to_img_id]

    # Now filter the dict to only these keys
    filtered_community_name_to_img_id = {k: community_name_to_img_id[k] for k in horse_community_name_list}
    
    # Now build the graph
    G = build_subset_graph(horse_community_name_list, filtered_community_name_to_img_id, trainsg_dupes=set(), subject_str=None)
    
    #G = build_subset_graph(horse_community_name_list, community_name_to_img_id, trainsg_dupes=set(), subject_str=None)

    spectral_pos = nx.spectral_layout(
        G=G, 
        dim=5,
        )
    
    for subset_A, subset_B in [
        ['horse(dirt)', 'horse(trees)'],
        ['horse(fence)', 'horse(helmet)'],
        ['horse(car)', 'horse(wagon)'],
        ['horse(statue)', 'horse(cart)'],
    ]:
        distance_A = np.linalg.norm(spectral_pos[subset_A.replace('(', '\n(')] - spectral_pos['horse\n(barn)'])
        distance_B = np.linalg.norm(spectral_pos[subset_B.replace('(', '\n(')] - spectral_pos['horse\n(barn)'])
        
        print('Distance from {}+{} to {}: {}'.format(
            subset_A, subset_B, 'horse(barn)', 
            0.5 * (distance_A + distance_B)
            )
        )

        
    return

if __name__ == '__main__':
    generate_splitted_metadaset()

