#!/usr/bin/env python3
"""
MetaShift Node2Vec Distance Measurement
=====================================

This script implements a node2vec approach to measure similarity between 
MetaShift subsets and compares it with the traditional spectral layout approach
used in domain_generalization_cat_dog.py.

The main idea is to compare MetaShift spectral distances with Node2Vec distances 
for the same subjects used in MetaShift domain generalization experiments.

Based on: domain_generalization_cat_dog.py
Author: MetaShift Analysis Team
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import pprint

# Node2Vec and ML libraries
from node2vec import Node2Vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr, spearmanr

# Import MetaShift utilities
import Constants
from generate_full_MetaShift import preprocess_groups, build_subset_graph, parse_node_str

# Set matplotlib backend for server environments
import matplotlib
matplotlib.use('Agg')

# Output folder
RESULTS_FOLDER = '../Node2Vec-Distance-Analysis'


class MetaShiftNode2VecComparison:
    """
    Compare MetaShift spectral layout distances with Node2Vec embedding distances
    using the exact same data and experimental setup as domain_generalization_cat_dog.py
    """
    
    def __init__(self):
        self.node_name_to_img_id = None
        self.train_set_scheme = None
        self.test_set_scheme = None
        self.additional_test_set_scheme = None
        self.trainsg_dupes = None
        self.results = {}
        
    def load_metashift_data(self):
        """Load MetaShift data exactly as in domain_generalization_cat_dog.py"""
        print("Loading MetaShift data...")
        
        # Load the same data as the original script
        self.node_name_to_img_id, most_common_list, subjects_to_all_set, subject_group_summary_dict = preprocess_groups(output_files_flag=False)
        
        print(f'node_name_to_img_id["dog(cabinet)"] length: {len(self.node_name_to_img_id["dog(cabinet)"])}')
        print(f'Total #object classes (subjects): {len(subject_group_summary_dict)}')
        print('Example object classes (subjects):', list(subject_group_summary_dict.keys())[:10])
        
        # Removing ambiguous images that have both cats and dogs (same as original)
        self.trainsg_dupes = self.node_name_to_img_id['cat(dog)']
        print(f"Excluded ambiguous images: {len(self.trainsg_dupes)}")
        
    def define_dataset_schemes(self):
        """Define the exact same train/test schemes as domain_generalization_cat_dog.py"""
        
        # Exact copy of the schemes from the original file
        self.train_set_scheme = {
            'cat': {
                'cat(sofa)': {'cat(pillow)', 'cat(sofa)', 'cat(wall)'}, 
                'cat(bed)':  {'cat(bed)', 'cat(comforter)', 'cat(sheet)', 'cat(blanket)', 'cat(wall)', 'cat(pillow)', 'cat(lamp)'}, 
            }, 
            'dog': {
                'dog(cabinet)': {'dog(wall)', 'dog(cabinet)', 'dog(laptop)', 'dog(door)', 'dog(chair)', 'dog(window)'}, 
                'dog(bed)': {'dog(blanket)', 'dog(bed)', 'dog(sheet)', 'dog(pillow)', 'dog(lamp)', 'dog(clothes)', 'dog(curtain)', 'dog(collar)', 'dog(wall)'}, 
                'dog(bag)': {'dog(bag)', 'dog(backpack)', 'dog(purse)'}, 
                'dog(box)': {'dog(box)', 'dog(container)', 'dog(food)', 'dog(table)', 'dog(plate)', 'dog(cup)'} , 
                'dog(bench)': {'dog(bench)', 'dog(trash can)'} , 
                'dog(bike)': {'dog(basket)', 'dog(woman)', 'dog(bike)', 'dog(bicycle)'}, 
                'dog(boat)': {'dog(frisbee)', 'dog(rope)', 'dog(flag)', 'dog(trees)', 'dog(boat)'}, 
                'dog(surfboard)': {'dog(water)', 'dog(surfboard)', 'dog(sand)'}, 
            }
        }

        self.test_set_scheme = {
            'cat': {
                'cat(shelf)': {'cat(container)', 'cat(shelf)', 'cat(vase)', 'cat(bowl)'}, 
            },
            'dog': {
                'dog(shelf)': {'dog(desk)', 'dog(screen)', 'dog(laptop)', 'dog(shelf)', 'dog(picture)', 'dog(chair)'}, 
            },
        }

        self.additional_test_set_scheme = {
            'cat': {
                'cat(grass)': {'cat(house)', 'cat(car)', 'cat(grass)', 'cat(bird)'},
                'cat(sink)': {'cat(sink)', 'cat(bottle)', 'cat(faucet)', 'cat(towel)', 'cat(toilet)'}, 
                'cat(computer)': {'cat(speaker)', 'cat(computer)', 'cat(screen)', 'cat(laptop)', 'cat(computer mouse)', 'cat(keyboard)', 'cat(monitor)', 'cat(desk)',}, 
                'cat(box)': {'cat(box)', 'cat(paper)', 'cat(suitcase)', 'cat(bag)',}, 
                'cat(book)': {'cat(books)', 'cat(book)', 'cat(television)', 'cat(bookshelf)', 'cat(blinds)',},
            },
            'dog': {
                'dog(sofa)': {'dog(sofa)', 'dog(television)', 'dog(carpet)',  'dog(phone)', 'dog(book)',}, 
                'dog(grass)': {'dog(house)', 'dog(grass)', 'dog(horse)', 'dog(cow)', 'dog(sheep)','dog(animal)'}, 
                'dog(vehicle)': {'dog(car)', 'dog(motorcycle)', 'dog(truck)', 'dog(bike)', 'dog(basket)', 'dog(bicycle)', 'dog(skateboard)', }, 
                'dog(cap)': {'dog(cap)', 'dog(scarf)', 'dog(jacket)', 'dog(toy)', 'dog(collar)', 'dog(tie)'},
            },
        }
        
        print("Dataset schemes defined (same as original domain_generalization_cat_dog.py)")
        
    def parse_dataset_scheme(self, dataset_scheme, exclude_img_id=set(), copy=False):
        """Parse dataset scheme - copied from original function"""
        community_name_to_img_id = defaultdict(set)
        all_img_id = set()

        for subject_str in dataset_scheme:        
            for community_name in dataset_scheme[subject_str]:
                for node_name in dataset_scheme[subject_str][community_name]:
                    community_name_to_img_id[community_name].update(
                        self.node_name_to_img_id[node_name] - exclude_img_id)
                    all_img_id.update(self.node_name_to_img_id[node_name] - exclude_img_id)
                if copy:
                    print(community_name, 'Size:', len(community_name_to_img_id[community_name]))

        return community_name_to_img_id, all_img_id
    
    def create_combined_graph(self, subject='dog'):
        """Create the same graph as in the original script for comparison"""
        print(f"\n=== Creating combined graph for {subject} ===")
        
        # Parse schemes exactly like the original - with correct exclusion logic
        print('========== Quantifying the distance between train and test subsets ==========')
        test_community_name_to_img_id, test_all_img_id = self.parse_dataset_scheme(
            self.test_set_scheme, exclude_img_id=self.trainsg_dupes, copy=False)
        
        train_community_name_to_img_id, train_all_img_id = self.parse_dataset_scheme(
            self.train_set_scheme, exclude_img_id=self.trainsg_dupes, copy=False)
        
        additional_test_community_name_to_img_id, additional_test_all_img_id = self.parse_dataset_scheme(
            self.additional_test_set_scheme, exclude_img_id=self.trainsg_dupes, copy=False)

        # Combine all community data - this is the key difference!
        community_name_to_img_id = test_community_name_to_img_id.copy()
        community_name_to_img_id.update(train_community_name_to_img_id)
        community_name_to_img_id.update(additional_test_community_name_to_img_id)
        
        # Create community list for the subject (same as original)
        if subject == 'dog':
            community_name_list = (sorted(self.train_set_scheme['dog']) + 
                                 sorted(self.test_set_scheme['dog']) + 
                                 sorted(self.additional_test_set_scheme['dog']))
        elif subject == 'cat':
            community_name_list = (sorted(self.train_set_scheme['cat']) + 
                                 sorted(self.test_set_scheme['cat']) + 
                                 sorted(self.additional_test_set_scheme['cat']))
        else:
            raise ValueError(f"Unknown subject: {subject}")
        
        print(f'Dog_community_name_list: {community_name_list}')
        print(f'len(Dog_community_name_list): {len(community_name_list)}')
        
        print(f'community_name_to_img_id keys: {list(community_name_to_img_id.keys())}')
        print(f'len(community_name_to_img_id): {len(community_name_to_img_id)}')
        
        # Build graph using MetaShift's function - KEY: use community data, not individual node data
        G = build_subset_graph(community_name_list, community_name_to_img_id, 
                              trainsg_dupes=set(), subject_str=None)
        
        print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G, community_name_to_img_id, community_name_list
    
    def compute_spectral_distances(self, G, reference_node='dog\n(shelf)', dim=5):
        """Compute spectral layout distances exactly as in the original script"""
        print("Computing spectral layout distances...")
        
        if G.number_of_nodes() < 2:
            print("Warning: Graph needs at least 2 nodes for spectral layout")
            return {}
            
        try:
            spectral_pos = nx.spectral_layout(G=G, dim=dim)
            print(f"Generated {dim}-dimensional spectral positions")
            
            # Calculate distances from reference node
            distances = {}
            if reference_node in spectral_pos:
                ref_pos = spectral_pos[reference_node]
                for node, pos in spectral_pos.items():
                    if node != reference_node:
                        distance = np.linalg.norm(pos - ref_pos)
                        distances[node] = distance
            
            return spectral_pos, distances
            
        except Exception as e:
            print(f"Error computing spectral layout: {e}")
            return {}, {}
    
    def compute_node2vec_embeddings(self, G, dimensions=64, walk_length=30, 
                                  num_walks=200, window=10, min_count=1):
        """Compute Node2Vec embeddings"""
        print("Computing Node2Vec embeddings...")
        
        if G.number_of_nodes() < 2:
            print("Warning: Graph needs at least 2 nodes for Node2Vec")
            return None
            
        try:
            # Create Node2Vec model
            node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, 
                              num_walks=num_walks, workers=1)
            
            # Train the model
            model = node2vec.fit(window=window, min_count=min_count, batch_words=4)
            
            print(f"Generated {dimensions}-dimensional embeddings for {len(model.wv.index_to_key)} nodes")
            return model
            
        except Exception as e:
            print(f"Error computing Node2Vec embeddings: {e}")
            return None
    
    def compute_node2vec_distances(self, model, G, reference_node='dog\n(shelf)'):
        """Compute distances in Node2Vec embedding space"""
        print("Computing Node2Vec distances...")
        
        if model is None:
            return {}
            
        try:
            distances = {}
            if reference_node in model.wv.index_to_key:
                ref_embedding = model.wv[reference_node]
                
                for node in G.nodes():
                    if node != reference_node and node in model.wv.index_to_key:
                        node_embedding = model.wv[node]
                        # Use Euclidean distance in embedding space
                        distance = euclidean(ref_embedding, node_embedding)
                        distances[node] = distance
                        
            return distances
            
        except Exception as e:
            print(f"Error computing Node2Vec distances: {e}")
            return {}
    
    def compare_distances(self, spectral_distances, node2vec_distances, subject='dog'):
        """Compare spectral and Node2Vec distances using the same pairs as original script"""
        print(f"\n=== Comparing distances for {subject} ===")
        
        # Use the exact same subset pairs as in the original script
        if subject == 'dog':
            subset_pairs = [
                ['dog(cabinet)', 'dog(bed)'],
                ['dog(bag)', 'dog(box)'],
                ['dog(bench)', 'dog(bike)'],
                ['dog(boat)', 'dog(surfboard)'],
            ]
            reference = 'dog(shelf)'
        elif subject == 'cat':
            subset_pairs = [
                ['cat(sofa)', 'cat(bed)'],
            ]
            reference = 'cat(shelf)'
        else:
            return {}
        
        results = []
        
        for subset_A, subset_B in subset_pairs:
            # Handle node naming (spectral layout adds \n for visualization)
            subset_A_spectral = subset_A.replace('(', '\n(')
            subset_B_spectral = subset_B.replace('(', '\n(')
            subset_A_node2vec = subset_A_spectral
            subset_B_node2vec = subset_B_spectral
            
            # Get spectral distances
            spectral_A = spectral_distances.get(subset_A_spectral, np.nan)
            spectral_B = spectral_distances.get(subset_B_spectral, np.nan)
            spectral_avg = 0.5 * (spectral_A + spectral_B) if not (np.isnan(spectral_A) or np.isnan(spectral_B)) else np.nan
            
            # Get Node2Vec distances
            node2vec_A = node2vec_distances.get(subset_A_node2vec, np.nan)
            node2vec_B = node2vec_distances.get(subset_B_node2vec, np.nan)
            node2vec_avg = 0.5 * (node2vec_A + node2vec_B) if not (np.isnan(node2vec_A) or np.isnan(node2vec_B)) else np.nan
            
            result = {
                'subset_pair': f"{subset_A} + {subset_B}",
                'reference': reference,
                'spectral_distance_A': spectral_A,
                'spectral_distance_B': spectral_B,
                'spectral_avg_distance': spectral_avg,
                'node2vec_distance_A': node2vec_A,
                'node2vec_distance_B': node2vec_B,
                'node2vec_avg_distance': node2vec_avg,
                'distance_ratio': node2vec_avg / spectral_avg if not np.isnan(spectral_avg) and spectral_avg != 0 else np.nan
            }
            
            results.append(result)
            
            # Print results in the same format as original script
            print(f'Distance from {subset_A}+{subset_B} to {reference}:')
            print(f'  Spectral: {spectral_avg:.4f}')
            print(f'  Node2Vec: {node2vec_avg:.4f}')
            print(f'  Ratio (N2V/Spectral): {result["distance_ratio"]:.4f}' if not np.isnan(result["distance_ratio"]) else '  Ratio: N/A')
            print()
        
        return results
    
    def analyze_all_pairwise_distances(self, spectral_distances, node2vec_distances):
        """Analyze correlation between spectral and Node2Vec distances across all pairs"""
        print("\n=== Analyzing all pairwise distance correlations ===")
        
        spectral_values = []
        node2vec_values = []
        node_pairs = []
        
        spectral_nodes = set(spectral_distances.keys())
        node2vec_nodes = set(node2vec_distances.keys())
        common_nodes = spectral_nodes.intersection(node2vec_nodes)
        
        print(f"Common nodes for comparison: {len(common_nodes)}")
        
        for node in common_nodes:
            if not np.isnan(spectral_distances[node]) and not np.isnan(node2vec_distances[node]):
                spectral_values.append(spectral_distances[node])
                node2vec_values.append(node2vec_distances[node])
                node_pairs.append(node)
        
        if len(spectral_values) < 2:
            print("Insufficient data for correlation analysis")
            return {}
        
        # Calculate correlations
        pearson_corr, pearson_p = pearsonr(spectral_values, node2vec_values)
        spearman_corr, spearman_p = spearmanr(spectral_values, node2vec_values)
        
        print(f"Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4f})")
        print(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4f})")
        
        correlation_results = {
            'n_pairs': len(spectral_values),
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'spectral_values': spectral_values,
            'node2vec_values': node2vec_values,
            'node_pairs': node_pairs
        }
        
        return correlation_results
    
    def visualize_original_graph(self, G, community_data, subject='dog', save_path=None):
        """
        Visualize the original MetaShift graph structure BEFORE Node2Vec processing.
        
        This visualization is important because it shows:
        1. The original semantic relationships in the dataset
        2. Edge weights representing image overlap coefficients
        3. Natural clustering based on visual contexts
        4. Baseline structure that Node2Vec will learn from
        
        Args:
            G: Original NetworkX graph
            community_data: Dictionary mapping community names to image sets
            subject: Subject being analyzed
            save_path: Path to save the visualization
        """
        print(f"Creating original graph visualization for {subject}...")
        
        try:
            # Create figure with detailed layout
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
            
            # 1. Spring Layout - Shows natural clustering
            ax1.set_title(f'Original Graph: Spring Layout - {subject}', fontsize=14, fontweight='bold')
            spring_pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
            
            # Color nodes by degree (connectivity)
            node_degrees = dict(G.degree())
            node_colors_degree = [node_degrees[node] for node in G.nodes()]
            
            nx.draw(G, pos=spring_pos, ax=ax1, with_labels=True,
                   node_color=node_colors_degree, cmap='viridis', 
                   node_size=1000, font_size=8, font_weight='bold',
                   edge_color='gray', alpha=0.7)
            
            # Add colorbar for node degrees
            sm = plt.cm.ScalarMappable(cmap='viridis', 
                                     norm=plt.Normalize(vmin=min(node_colors_degree), 
                                                       vmax=max(node_colors_degree)))
            sm.set_array([])
            cbar1 = plt.colorbar(sm, ax=ax1, shrink=0.8)
            cbar1.set_label('Node Degree (Connectivity)', rotation=270, labelpad=20)
            
            # 2. Circular Layout - Shows all connections clearly
            ax2.set_title(f'Original Graph: Circular Layout - {subject}', fontsize=14, fontweight='bold')
            circular_pos = nx.circular_layout(G)
            
            # Color nodes by community size
            community_sizes = {}
            for node in G.nodes():
                community_sizes[node] = len(community_data.get(node, set()))
            
            node_colors_size = [community_sizes[node] for node in G.nodes()]
            
            nx.draw(G, pos=circular_pos, ax=ax2, with_labels=True,
                   node_color=node_colors_size, cmap='plasma',
                   node_size=1200, font_size=8, font_weight='bold',
                   edge_color='lightblue', alpha=0.6)
            
            # Add colorbar for community sizes
            sm2 = plt.cm.ScalarMappable(cmap='plasma', 
                                      norm=plt.Normalize(vmin=min(node_colors_size), 
                                                        vmax=max(node_colors_size)))
            sm2.set_array([])
            cbar2 = plt.colorbar(sm2, ax=ax2, shrink=0.8)
            cbar2.set_label('Community Size (# Images)', rotation=270, labelpad=20)
            
            # 3. Edge Weight Distribution
            ax3.set_title(f'Edge Weight Distribution - {subject}', fontsize=14, fontweight='bold')
            edge_weights = [G[u][v].get('weight', 0) for u, v in G.edges()]
            
            if edge_weights:
                ax3.hist(edge_weights, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax3.axvline(np.mean(edge_weights), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(edge_weights):.3f}')
                ax3.axvline(np.median(edge_weights), color='green', linestyle='--', 
                           label=f'Median: {np.median(edge_weights):.3f}')
                ax3.set_xlabel('Edge Weight (Overlap Coefficient)')
                ax3.set_ylabel('Frequency')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # 4. Network Statistics
            ax4.set_title(f'Network Statistics - {subject}', fontsize=14, fontweight='bold')
            ax4.axis('off')
            
            # Calculate network metrics
            stats_text = []
            stats_text.append(f"Nodes: {G.number_of_nodes()}")
            stats_text.append(f"Edges: {G.number_of_edges()}")
            stats_text.append(f"Density: {nx.density(G):.3f}")
            stats_text.append(f"Average Clustering: {nx.average_clustering(G):.3f}")
            
            if nx.is_connected(G):
                stats_text.append(f"Average Path Length: {nx.average_shortest_path_length(G):.3f}")
                stats_text.append(f"Diameter: {nx.diameter(G)}")
            else:
                stats_text.append("Graph is not connected")
                
            stats_text.append(f"Average Degree: {np.mean(list(dict(G.degree()).values())):.2f}")
            
            if edge_weights:
                stats_text.append(f"Mean Edge Weight: {np.mean(edge_weights):.3f}")
                stats_text.append(f"Std Edge Weight: {np.std(edge_weights):.3f}")
            
            # Display statistics
            y_pos = 0.9
            for stat in stats_text:
                ax4.text(0.1, y_pos, stat, transform=ax4.transAxes, fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
                y_pos -= 0.1
            
            plt.tight_layout()
            
            # Save the plot
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Original graph visualization saved to {save_path}")
            else:
                default_path = f'{subject}_original_graph_analysis.png'
                plt.savefig(default_path, dpi=300, bbox_inches='tight')
                print(f"Original graph visualization saved to {default_path}")
                
            plt.close()
            
        except Exception as e:
            print(f"Error creating original graph visualization: {e}")
            import traceback
            traceback.print_exc()

    def visualize_node2vec_embedding_analysis(self, model, G, subject='dog', save_path=None):
        """
        Visualize Node2Vec embeddings and their properties AFTER processing.
        
        This visualization is important because it shows:
        1. How Node2Vec transforms graph structure into vector space
        2. Clustering patterns in embedding space
        3. Dimensionality reduction reveals semantic groups
        4. Distance relationships in learned representation
        
        Args:
            model: Trained Node2Vec model
            G: Original graph structure
            subject: Subject being analyzed
            save_path: Path to save the visualization
        """
        print(f"Creating Node2Vec embedding analysis for {subject}...")
        
        if model is None:
            print("No Node2Vec model available for visualization")
            return
            
        try:
            # Get embeddings
            nodes = [node for node in G.nodes() if node in model.wv.index_to_key]
            embeddings = np.array([model.wv[node] for node in nodes])
            
            if len(nodes) < 2:
                print("Insufficient nodes for embedding analysis")
                return
            
            # Create comprehensive figure
            fig = plt.figure(figsize=(20, 16))
            
            # 1. t-SNE 2D Projection
            ax1 = plt.subplot(2, 3, 1)
            perplexity = min(5, len(nodes) - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            embeddings_2d = tsne.fit_transform(embeddings)
            
            # Color by original graph degree
            node_degrees = [G.degree(node) for node in nodes]
            scatter1 = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                  c=node_degrees, cmap='viridis', s=100, alpha=0.7)
            
            # Add labels
            for i, node in enumerate(nodes):
                ax1.annotate(node.replace('\n', ' '), 
                           (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.8)
            
            ax1.set_title('t-SNE: Node2Vec Embeddings by Graph Degree', fontweight='bold')
            plt.colorbar(scatter1, ax=ax1, label='Node Degree')
            
            # 2. PCA 2D Projection
            ax2 = plt.subplot(2, 3, 2)
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            embeddings_pca = pca.fit_transform(embeddings)
            
            scatter2 = ax2.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                                  c=node_degrees, cmap='plasma', s=100, alpha=0.7)
            
            for i, node in enumerate(nodes):
                ax2.annotate(node.replace('\n', ' '), 
                           (embeddings_pca[i, 0], embeddings_pca[i, 1]),
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.8)
            
            ax2.set_title(f'PCA: Node2Vec Embeddings\n(Explained Var: {pca.explained_variance_ratio_.sum():.2f})', 
                         fontweight='bold')
            plt.colorbar(scatter2, ax=ax2, label='Node Degree')
            
            # 3. Embedding Distance Heatmap
            ax3 = plt.subplot(2, 3, 3)
            distance_matrix = np.zeros((len(nodes), len(nodes)))
            
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    distance_matrix[i, j] = euclidean(embeddings[i], embeddings[j])
            
            im = ax3.imshow(distance_matrix, cmap='coolwarm', aspect='auto')
            ax3.set_xticks(range(len(nodes)))
            ax3.set_yticks(range(len(nodes)))
            ax3.set_xticklabels([node.replace('\n', ' ') for node in nodes], rotation=45, ha='right')
            ax3.set_yticklabels([node.replace('\n', ' ') for node in nodes])
            ax3.set_title('Node2Vec Distance Matrix', fontweight='bold')
            plt.colorbar(im, ax=ax3, label='Euclidean Distance')
            
            # 4. Embedding Dimension Analysis
            ax4 = plt.subplot(2, 3, 4)
            embedding_norms = np.linalg.norm(embeddings, axis=1)
            ax4.bar(range(len(nodes)), embedding_norms, alpha=0.7, color='lightcoral')
            ax4.set_xticks(range(len(nodes)))
            ax4.set_xticklabels([node.replace('\n', ' ') for node in nodes], rotation=45, ha='right')
            ax4.set_ylabel('Embedding Norm')
            ax4.set_title('Node2Vec Embedding Magnitudes', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # 5. Similarity Network in Embedding Space
            ax5 = plt.subplot(2, 3, 5)
            # Create similarity graph based on embedding distances
            similarity_threshold = np.percentile(distance_matrix[distance_matrix > 0], 30)  # Top 70% similar
            similarity_graph = nx.Graph()
            similarity_graph.add_nodes_from(nodes)
            
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if distance_matrix[i, j] < similarity_threshold:
                        similarity_graph.add_edge(nodes[i], nodes[j], 
                                                weight=1.0 / (distance_matrix[i, j] + 1e-6))
            
            pos_sim = nx.spring_layout(similarity_graph, k=1, iterations=50)
            nx.draw(similarity_graph, pos=pos_sim, ax=ax5, with_labels=True,
                   node_color='lightgreen', node_size=800, font_size=8,
                   edge_color='gray', alpha=0.7)
            ax5.set_title('Similarity Network\n(Based on Node2Vec Embeddings)', fontweight='bold')
            
            # 6. Embedding Statistics
            ax6 = plt.subplot(2, 3, 6)
            ax6.axis('off')
            
            # Calculate embedding statistics
            stats_text = []
            stats_text.append(f"Embedding Dimension: {embeddings.shape[1]}")
            stats_text.append(f"Number of Nodes: {len(nodes)}")
            stats_text.append(f"Mean Embedding Norm: {np.mean(embedding_norms):.3f}")
            stats_text.append(f"Std Embedding Norm: {np.std(embedding_norms):.3f}")
            stats_text.append(f"Mean Pairwise Distance: {np.mean(distance_matrix[distance_matrix > 0]):.3f}")
            stats_text.append(f"Std Pairwise Distance: {np.std(distance_matrix[distance_matrix > 0]):.3f}")
            
            # PCA explained variance
            full_pca = PCA().fit(embeddings)
            cumsum_var = np.cumsum(full_pca.explained_variance_ratio_)
            dims_90 = np.argmax(cumsum_var >= 0.9) + 1
            dims_95 = np.argmax(cumsum_var >= 0.95) + 1
            
            stats_text.append(f"Dims for 90% variance: {dims_90}")
            stats_text.append(f"Dims for 95% variance: {dims_95}")
            
            # Display statistics
            y_pos = 0.9
            for stat in stats_text:
                ax6.text(0.1, y_pos, stat, transform=ax6.transAxes, fontsize=11, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5))
                y_pos -= 0.11
            
            ax6.set_title('Node2Vec Embedding Statistics', fontweight='bold', y=0.95)
            
            plt.tight_layout()
            
            # Save the plot
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Node2Vec embedding analysis saved to {save_path}")
            else:
                default_path = f'{subject}_node2vec_embedding_analysis.png'
                plt.savefig(default_path, dpi=300, bbox_inches='tight')
                print(f"Node2Vec embedding analysis saved to {default_path}")
                
            plt.close()
            
        except Exception as e:
            print(f"Error creating Node2Vec embedding analysis: {e}")
            import traceback
            traceback.print_exc()

    def visualize_comparison(self, spectral_pos, model, G, subject='dog', save_path=None):
        """Create visualization comparing spectral layout vs Node2Vec embeddings"""
        print(f"Creating visualization for {subject}...")
        
        if not spectral_pos or model is None:
            print("Missing data for visualization")
            return
            
        try:
            # Get Node2Vec 2D positions using t-SNE
            nodes = list(G.nodes())
            embeddings = np.array([model.wv[node] for node in nodes if node in model.wv.index_to_key])
            valid_nodes = [node for node in nodes if node in model.wv.index_to_key]
            
            if len(valid_nodes) < 2:
                print("Insufficient nodes for t-SNE visualization")
                return
                
            # Use t-SNE for 2D projection of Node2Vec embeddings
            perplexity = min(5, len(valid_nodes) - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            embeddings_2d = tsne.fit_transform(embeddings)
            
            node2vec_pos = {node: pos for node, pos in zip(valid_nodes, embeddings_2d)}
            
            # Create 2D positions from spectral layout
            spectral_pos_2d = {}
            if spectral_pos:
                # For spectral layout, we need to project 5D to 2D for visualization
                spectral_embeddings = np.array([pos for pos in spectral_pos.values()])
                spectral_nodes = list(spectral_pos.keys())
                
                if len(spectral_nodes) >= 2:
                    # Use t-SNE to project spectral embeddings to 2D
                    spectral_perplexity = min(5, len(spectral_nodes) - 1)
                    spectral_tsne = TSNE(n_components=2, random_state=42, perplexity=spectral_perplexity)
                    spectral_2d = spectral_tsne.fit_transform(spectral_embeddings)
                    spectral_pos_2d = {node: pos for node, pos in zip(spectral_nodes, spectral_2d)}
                elif len(spectral_nodes) == 1:
                    # If only one node, place it at origin
                    spectral_pos_2d = {spectral_nodes[0]: np.array([0, 0])}
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # Plot spectral layout
            ax1.set_title(f'Spectral Layout (5D→2D via t-SNE) - {subject}', fontsize=16)
            if spectral_pos_2d:
                nx.draw(G, pos=spectral_pos_2d, ax=ax1, with_labels=True, 
                       node_color='lightblue', node_size=1200, font_size=8, 
                       font_weight='bold', edge_color='gray', alpha=0.7)
            else:
                ax1.text(0.5, 0.5, 'No spectral layout data', ha='center', va='center', 
                        transform=ax1.transAxes, fontsize=12)
            
            # Plot Node2Vec layout
            ax2.set_title(f'Node2Vec + t-SNE - {subject}', fontsize=16)
            if node2vec_pos:
                nx.draw(G, pos=node2vec_pos, ax=ax2, with_labels=True,
                       node_color='lightcoral', node_size=1200, font_size=8,
                       font_weight='bold', edge_color='gray', alpha=0.7)
            else:
                ax2.text(0.5, 0.5, 'No Node2Vec data', ha='center', va='center',
                        transform=ax2.transAxes, fontsize=12)
            
            plt.tight_layout()
            
            # Ensure the output directory exists
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to {save_path}")
            else:
                default_path = f'{subject}_spectral_vs_node2vec_comparison.png'
                plt.savefig(default_path, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to {default_path}")
                
            plt.close()
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def save_results(self, results, correlation_results, subject='dog'):
        """Save detailed results to files"""
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        
        # Save comparison results
        if results:
            results_df = pd.DataFrame(results)
            results_path = os.path.join(RESULTS_FOLDER, f'{subject}_distance_comparison.csv')
            results_df.to_csv(results_path, index=False)
            print(f"Distance comparison results saved to {results_path}")
        
        # Save correlation results
        if correlation_results:
            corr_path = os.path.join(RESULTS_FOLDER, f'{subject}_correlation_analysis.txt')
            with open(corr_path, 'w') as f:
                f.write(f"MetaShift Node2Vec vs Spectral Distance Correlation Analysis - {subject}\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Number of distance pairs: {correlation_results['n_pairs']}\n")
                f.write(f"Pearson correlation: {correlation_results['pearson_correlation']:.4f} ")
                f.write(f"(p-value: {correlation_results['pearson_p_value']:.4f})\n")
                f.write(f"Spearman correlation: {correlation_results['spearman_correlation']:.4f} ")
                f.write(f"(p-value: {correlation_results['spearman_p_value']:.4f})\n\n")
                
                f.write("Individual distance pairs:\n")
                f.write("-" * 40 + "\n")
                for i, node in enumerate(correlation_results['node_pairs']):
                    f.write(f"{node}: Spectral={correlation_results['spectral_values'][i]:.4f}, ")
                    f.write(f"Node2Vec={correlation_results['node2vec_values'][i]:.4f}\n")
            
            print(f"Correlation analysis saved to {corr_path}")
    
    def run_full_comparison(self, subjects=['dog', 'cat']):
        """Run the complete comparison analysis"""
        print("=" * 70)
        print("MetaShift Node2Vec vs Spectral Distance Comparison")
        print("=" * 70)
        
        # Load data and define schemes
        self.load_metashift_data()
        self.define_dataset_schemes()
        
        for subject in subjects:
            print(f"\n{'='*20} ANALYZING {subject.upper()} {'='*20}")
            
            try:
                # Create graph
                G, community_data, community_list = self.create_combined_graph(subject)
                
                if G.number_of_nodes() < 2:
                    print(f"Skipping {subject} - insufficient nodes in graph")
                    continue
                
                # === NEW: Visualize original graph structure ===
                original_viz_path = os.path.join(RESULTS_FOLDER, f'{subject}_original_graph_analysis.png')
                self.visualize_original_graph(G, community_data, subject, original_viz_path)
                
                # Compute spectral distances
                spectral_pos, spectral_distances = self.compute_spectral_distances(
                    G, reference_node=f'{subject}\n(shelf)')
                
                # Compute Node2Vec embeddings and distances
                model = self.compute_node2vec_embeddings(G)
                node2vec_distances = self.compute_node2vec_distances(
                    model, G, reference_node=f'{subject}\n(shelf)')
                
                # === NEW: Visualize Node2Vec embedding analysis ===
                embedding_viz_path = os.path.join(RESULTS_FOLDER, f'{subject}_node2vec_embedding_analysis.png')
                self.visualize_node2vec_embedding_analysis(model, G, subject, embedding_viz_path)
                
                # Compare distances for specific pairs
                comparison_results = self.compare_distances(
                    spectral_distances, node2vec_distances, subject)
                
                # Analyze all pairwise correlations
                correlation_results = self.analyze_all_pairwise_distances(
                    spectral_distances, node2vec_distances)
                
                # Create final comparison visualization
                viz_path = os.path.join(RESULTS_FOLDER, f'{subject}_layout_comparison.png')
                self.visualize_comparison(spectral_pos, model, G, subject, viz_path)
                
                # Save results
                self.save_results(comparison_results, correlation_results, subject)
                
                # Store results for summary
                self.results[subject] = {
                    'graph_nodes': G.number_of_nodes(),
                    'graph_edges': G.number_of_edges(),
                    'comparison_results': comparison_results,
                    'correlation_results': correlation_results
                }
                
            except Exception as e:
                print(f"Error processing {subject}: {e}")
                continue
        
        # Create summary
        self.create_summary_report()
        
        print(f"\n{'='*20} ANALYSIS COMPLETE {'='*20}")
        print(f"Results saved to: {RESULTS_FOLDER}")
    
    def create_summary_report(self):
        """Create a summary report of all results"""
        summary_path = os.path.join(RESULTS_FOLDER, 'summary_report.txt')
        
        with open(summary_path, 'w') as f:
            f.write("MetaShift Node2Vec vs Spectral Distance Analysis - Summary Report\n")
            f.write("=" * 80 + "\n\n")
            
            for subject, data in self.results.items():
                f.write(f"Subject: {subject.upper()}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Graph nodes: {data['graph_nodes']}\n")
                f.write(f"Graph edges: {data['graph_edges']}\n")
                
                if data['correlation_results']:
                    corr_data = data['correlation_results']
                    f.write(f"Distance pairs analyzed: {corr_data['n_pairs']}\n")
                    f.write(f"Pearson correlation: {corr_data['pearson_correlation']:.4f}\n")
                    f.write(f"Spearman correlation: {corr_data['spearman_correlation']:.4f}\n")
                
                if data['comparison_results']:
                    f.write(f"Specific subset pair comparisons: {len(data['comparison_results'])}\n")
                    for result in data['comparison_results']:
                        if not np.isnan(result['distance_ratio']):
                            f.write(f"  {result['subset_pair']}: Ratio = {result['distance_ratio']:.4f}\n")
                
                f.write("\n")
        
        print(f"Summary report saved to {summary_path}")


def main():
    """Main execution function"""
    # Create comparison analyzer
    analyzer = MetaShiftNode2VecComparison()
    
    # Run complete analysis
    analyzer.run_full_comparison(subjects=['dog'])  # Start with dog, add 'cat' if needed
    
    print("\nNode2Vec vs Spectral Distance comparison completed!")
    print("Check the results folder for detailed analysis.")


if __name__ == "__main__":
    main()