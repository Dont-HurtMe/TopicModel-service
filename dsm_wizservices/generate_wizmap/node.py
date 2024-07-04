import re
import os
import shutil
import pickle
import json  
import torch
import ndjson
import pandas as pd
import numpy as np
import random
import subprocess

from pathlib import Path
from tqdm import tqdm
from json import dump
from glob import glob
from os.path import exists, join, basename
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KernelDensity
from typing import List, Tuple, Union, Mapping, Any, Callable, Iterable, Dict, Literal
from quadtreed3 import Quadtree, Node
from scipy.sparse import csr_matrix

# data_dir = "../../data_store"
class Generate_Wizmap:

    def __init__(self, 
                 save_file_path: Path, 
                 cv: CountVectorizer = None, 
                 verbose: bool = True):

        """
        Args:
            save_file_path: The path where the file should be saved.
            cv: An optional CountVectorizer object.
            verbose: Whether to print verbose output.
        """

        self.cv = cv
        self.SEED = random.randint(0, 2**32 - 1)
        self.verbose = verbose
        self.save_file_path = save_file_path

    def input(self, data :pd.DataFrame):

        """
            data: The DataFrame containing the data. But data should be follow this pattern.
                            _______________________________________________
            data pattern : | x | y | data_process | data_display | vector |
                            -----------------------------------------------   
        """

        self.data =  data 
        
    def NDJsonData(self, 
                   data: pd.DataFrame) -> List:
                   # labels: List[int] = None

        texts = list(data['data_display'])
        xs = list(data['x'])
        ys = list(data['y'])
        
        dataNDJson = []
        for i in tqdm(range(len(xs))):
            cur_row = [xs[i], ys[i], texts[i]]
            # if labels is not None:
            #     cur_row.append('')
            #     cur_row.append(labels[i])
            dataNDJson.append(cur_row)

        return dataNDJson

    def JsonGrid(self, 
                 data: pd.DataFrame, 
                 length_of_data: int) -> dict:
        
        texts = data['data_process'].tolist()
        xs = list(data['x'])
        ys = list(data['y'])
        x_min, x_max, y_min, y_max, grid_density, sample_size = self.bandwidth_silverman(xs, ys)
        grid_density_json = {
                                'grid': grid_density.astype(float).round(4).tolist(),
                                'xRange': [x_min, x_max],
                                'yRange': [y_min, y_max],
                                'sampleSize': sample_size,
                                'totalPointSize': length_of_data,
                                'padded': True,
                            }
        
        data_dict = self.GridSummary(xs, ys, texts)
        topic = {'topic': data_dict}
        grid_dict = {**grid_density_json, **topic}
        
        return grid_dict    
        
    def GridSummary(self, 
                    xs: List[Union[float, int]], 
                    ys: List[Union[float, int]], 
                    texts: List[Union[float, int, str]]) -> dict: 

        cv = self.cv
        
        # Create data array
        data = []
        for i in range(len(xs)):
            cur_data = {
                'x': xs[i],
                'y': ys[i],
                'pid': i,
            }
            data.append(cur_data)
    
        # Build the quadtree
        tree = Quadtree()
        tree.add_all_data(data)
        
        # Build the count matrix
        root = tree.get_node_representation()
        
        count_mat = cv.fit_transform(texts)
        ngrams = cv.get_feature_names_out()

        max_zoom_scale = 20
        svg_width = 800
        svg_height = 800
        xs = [d['x'] for d in data]
        ys = [d['y'] for d in data]
        x_domain = [np.min(xs), np.max(xs)]
        y_domain = [np.min(ys), np.max(ys)]
        
        # Get suggestions of quadtree levels to extract
        min_level, max_level = select_topic_levels(
            max_zoom_scale, svg_width, svg_height, x_domain, y_domain, tree.extent()
        )

        level_tile_topics = extract_level_topics(
            root, count_mat, texts, ngrams, min_level=min_level, max_level=max_level
        )
        
        data_dict = {
            'extent': tree.extent(),
            'data': {},
            'range': [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
        }
        
        for cur_level in range(min_level, max_level + 1):
            cur_topics = level_tile_topics[cur_level]
            data_dict['data'][cur_level] = []
        
            for topic in cur_topics:
                # Get the topic name
                name = '-'.join([p[0] for p in topic['w'][:4]])
                x = (topic['p'][0] + topic['p'][2]) / 2
                y = (topic['p'][1] + topic['p'][3]) / 2
                cur_data = {
                    'x': round(x, 3),
                    'y': round(y, 3),
                    'n': name,
                    'l': cur_level
                }
                data_dict['data'][cur_level].append(
                    [round(x, 3), round(y, 3), name]
                )

        return data_dict

    def get_coord(self, 
                  xs: List[Union[float, int]], 
                  ys: List[Union[float, int]]) -> List:
        
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        
        x_gap = x_max - x_min
        y_gap = y_max - y_min
        
        if x_gap > y_gap:
            # Expand the larger range to leave some padding in the plots
            x_min -= x_gap / 50
            x_max += x_gap / 50
            x_gap = x_max - x_min
            
            # Regulate the 2D grid to be a square
            y_min -= (x_gap - y_gap) / 2
            y_max += (x_gap - y_gap) / 2
        else:
            # Expand the larger range to leave some padding in the plots
            y_min -= y_gap / 50
            y_max += y_gap / 50
            y_gap = y_max - y_min
            
            # Regulate the 2D grid to be a square
            x_min -= (y_gap - x_gap) / 2
            x_max += (y_gap - x_gap) / 2
        
        # Estimate on a 2D grid
        grid_size = 200
        grid_xs = np.linspace(x_min, x_max, grid_size)
        grid_ys = np.linspace(y_min, y_max, grid_size)
        xx, yy = np.meshgrid(grid_xs, grid_ys)
        
        grid = np.vstack([xx.ravel(), yy.ravel()]).transpose()
        return [grid, x_min, y_min, x_max, y_max, xx, yy]

    def bandwidth_silverman (self, 
                             xs: List[Union[float, int]], 
                             ys: List[Union[float, int]]):

        projected_emb = np.stack((xs, ys), axis=1)
        grid, x_min, y_min, x_max, y_max, xx, yy = self.get_coord(xs, ys)

        # # Compute the bandwidth using silverman's rule
        sample_size = 100000
        n = sample_size
        d = projected_emb.shape[1]
        bw = (n * (d + 2) / 4.)**(-1. / (d + 4))
        
        # We use a random sample to fit the KDE for faster run time
        rng = np.random.RandomState(self.SEED)
        random_indexes = rng.choice(range(projected_emb.shape[0]),
                                    min(projected_emb.shape[0], sample_size),
                                    replace=False)
        
        kde = KernelDensity(kernel='gaussian', bandwidth=bw)
        kde.fit(projected_emb[random_indexes, :])
        
        log_density = kde.score_samples(grid)
        log_density = np.exp(log_density)
        grid_density = np.reshape(log_density, xx.shape)
        grid_density.shape
            
        x_min, x_max, y_min, y_max = float(x_min), float(x_max), float(y_min), float(y_max)

        if self.verbose:

            plt.figure(figsize=(12,6))
    
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

            # Contourf plot
            plt.title(f'KDE on {grid_density.shape[0]} Grid of {sample_size} Samples (bw={bw:.2f})')
            plt.contourf(xx, yy, grid_density.round(4), levels=np.linspace(0, np.max(grid_density), 20), cmap=plt.cm.Blues)
            plt.colorbar()
            plt.scatter(xs, ys, alpha=0.2, s=0.1)
            
            if self.save_file_path.endswith('.pkl'):
                file_path = self.save_file_path
            else:
                file_path = os.path.join(self.save_file_path, 'model.pkl')

            with open(os.path.join(file_path), 'rb') as f:
                loaded_object = pickle.load(f)

            try : 
                cluster_model = loaded_object.get('clustering')
                centers = cluster_model.cluster_centers_
                cluster_model = None
                plt.scatter(centers[:, 0], centers[:, 1], marker='x', c='red', s=50, edgecolors='k')
            except :
                centers = calculate_cluster_centers(self.data[['x','y','cluster']])
                plt.scatter(centers[:, 0], centers[:, 1], marker='x', c='red', s=50, edgecolors='k')

            plt.xlabel('UMAP Dimension 1')
            plt.ylabel('UMAP Dimension 2')
            plt.show() 
            
            plt.figure(figsize=(12,6))
            cluster_list = self.data['cluster'].tolist()
            colors = [plt.cm.viridis(i / len(cluster_list)) for i in range(len(cluster_list))]
            plt.scatter(xs, ys, alpha=0.9, s=0.1, c=cluster_list)
            plt.title(f'Plot Labels')
            plt.show() 

        return x_min, x_max, y_min, y_max, grid_density, sample_size 

    def run(self, ):

        grid_dict = self.JsonGrid(data=self.data, length_of_data = self.data.shape[0])
        data_dict = self.NDJsonData(self.data)
        class NpEncoder(json.JSONEncoder):
            
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)

        base_save_dir = os.path.join(self.save_file_path, 'wizmapJson')
        os.makedirs(base_save_dir, exist_ok=True)

        with open(join(base_save_dir, 'data.ndjson'), "w", encoding="utf-8") as ndjson_file:
            for item in data_dict:
                ndjson_file.write(json.dumps(item, ensure_ascii=False, cls=NpEncoder) + "\n")
        
        with open(join(base_save_dir, 'grid.json'), 'w', encoding='utf-8') as file:
            json.dump(grid_dict, file, ensure_ascii=False) 


def top_n_idx_sparse(matrix: csr_matrix, n: int) -> np.ndarray:
    
    """ Return indices of top n values in each row of a sparse matrix
    Retrieved from:
        https://github.com/MaartenGr/BERTopic/blob/master/bertopic/_bertopic.py#L2801
    Arguments:
        matrix: The sparse matrix from which to get the top n indices per row
        n: The number of highest values to extract from each row
    Returns:
        indices: The top n indices per row
    """
    
    indices = []

    for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        n_row_pick = min(n, ri - le)
        values = matrix.indices[le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]]
        values = [values[index] if len(values) >= index + 1 else None for index in range(n)]
        indices.append(values)

    return np.array(indices)

def top_n_values_sparse(matrix: csr_matrix, indices: np.ndarray) -> np.ndarray:
    
    """ Return the top n values for each row in a sparse matrix
    Arguments:
        matrix: The sparse matrix from which to get the top n indices per row
        indices: The top n indices per row
    Returns:
        top_values: The top n scores per row
    """
    
    top_values = []
    for row in range(indices.shape[0]):
        scores = np.array([matrix[row, c] if c is not None else 0 for c in indices[row, :]])
        top_values.append(scores)
    return np.array(top_values)

def get_tile_topics(count_mat, row_pos_map, ngrams, top_k=10):
    
    """Get the top-k important keywords from all rows in the count_mat.

    Args:
        count_mat (csr_mat): A count matrix
        row_pos_map (dict): A dictionary that maps row index to the corresponding
            leaf node's location in the quadtree
        ngrams (list[str]): Feature names in the count_mat
        top_k (int): Number of keywords to extract
    """
    
    # Compute tf-idf score
    t_tf_idf_model = TfidfTransformer()
    t_tf_idf = t_tf_idf_model.fit_transform(count_mat)
    
    # Get words with top scores for each tile
    indices = top_n_idx_sparse(t_tf_idf, top_k)
    scores = top_n_values_sparse(t_tf_idf, indices)

    sorted_indices = np.argsort(scores, 1)
    indices = np.take_along_axis(indices, sorted_indices, axis=1)
    scores = np.take_along_axis(scores, sorted_indices, axis=1)
    
    # Store these keywords
    tile_topics = []

    for r in row_pos_map:
        word_scores = [
            (ngrams[word_index], round(score, 4))
            if word_index is not None and score > 0
            else ("", 0.00001)
            for word_index, score in zip(indices[r][::-1], scores[r][::-1])
        ]

        tile_topics.append({
            'w': word_scores,
            'p': row_pos_map[r]
        })

    return tile_topics

def extract_level_topics(
    root: Node,
    count_mat: csr_matrix,
    texts: list[str],
    ngrams: list[str],
    min_level = None,
    max_level = None
):
    
    """Extract topics for all leaf nodes at all levels of the quadtree.

    Args:
        root (Noe): Quadtree node
        count_mat (csr_matrix): Count vector for the corpus
        texts (list[str]): A list of all the embeddings' texts
        ngrams (list[str]): n-gram list for the count vectorizer
    """

    level_tile_topics = {}
    
    if min_level is None:
        min_level = 0

    if max_level is None:
        max_level = root.height

    for level in tqdm(list(range(max_level, min_level - 1, -1))):

        # Create a sparse matrix
        csr_row_indexes, csr_column_indexes, row_node_map = merge_leaves_before_level(
            root, level
        )

        csr_data = [1 for _ in range(len(csr_row_indexes))]
        tile_mat = csr_matrix(
            (csr_data, (csr_row_indexes, csr_column_indexes)),
            shape=(len(texts), len(texts)),
        )

        # Transform the count matrix
        new_count_mat = tile_mat @ count_mat

        # Compute t-tf-idf scores and extract keywords
        tile_topics = get_tile_topics(new_count_mat, row_node_map, ngrams)
        
        level_tile_topics[level] = tile_topics
        
    return level_tile_topics

def select_topic_levels(
    max_zoom_scale,
    svg_width,
    svg_height,
    x_domain,
    y_domain,
    tree_extent,
    ideal_tile_width=35):
    
    """
    Automatically determine the min and max topic levels needed for the visualization.

    Args:
        max_zoom_scale (float): Max zoom scale level
        svg_width (int): SVG width
        svg_height (int): SVG height
        x_domain ([float, float]): [x min, x max]
        y_domain ([float, float]): [y min, y max]
        tree_extent ([[float, float], [float, float]]): The extent of the tree
        ideal_tile_width (int, optional): Optimal tile width in pixel. Defaults to 35.
    """

    svg_length = max(svg_width, svg_height)
    world_length = max(x_domain[1] - x_domain[0], y_domain[1] - y_domain[0])
    tree_to_world_scale = (tree_extent[1][0] - tree_extent[0][0]) / world_length

    scale = 1
    selected_levels = []

    while scale <= max_zoom_scale:
        best_level = 1
        best_tile_width_diff = np.Infinity

        for l in range(1, 21):
            tile_num = 2**l
            svg_scaled_length = scale * svg_length * tree_to_world_scale
            tile_width = svg_scaled_length / tile_num

            if abs(tile_width - ideal_tile_width) < best_tile_width_diff:
                best_tile_width_diff = abs(tile_width - ideal_tile_width)
                best_level = l

        selected_levels.append(best_level)
        scale += 0.5

    return np.min(selected_levels), np.max(selected_levels)

def merge_leaves_before_level(root: Node, target_level: int) -> Tuple[list, list, dict]:
    
    """
    Merge all nodes to their parents until the tree is target_level tall (modify
    root in-place) and extract all data from leaf nodes before or at the target_level.

    Args:
        root (Node): Root node
        target_level (int): Target level

    Returns:
        csr_row_indexes (list): Row indexes for the sparse matrix. Each row is
            a leaf node.
        csr_column_indexes (list): Column indexes for the sparse matrix. Each column
            is a prompt ID.
        row_node_map (dict): A dictionary map row index to the leaf node.
    """
    
    x0, y0, x1, y1 = root.position
    step_size = (x1 - x0) / (2 ** target_level)

    # Find all leaves at or before the target level
    row_pos_map = {}
    stack = [root]

    # We create a sparse matrix by (data, (row index, column index))
    csr_row_indexes, csr_column_indexes = [], []

    # In the multiplication sparse matrix, each row represents a tile / collection,
    # and each column represents a prompt ID
    cur_r = 0

    while len(stack) > 0:
        cur_node = stack.pop()
        
        if cur_node.level >= target_level:        
            # A new traverse here to concatenate all the prompts from its subtree,
            # and to merge it with its children
            local_stack = [cur_node]
            subtree_data = []

            while len(local_stack) > 0:
                local_node = local_stack.pop()
                
                if len(local_node.children) == 0:
                    # Leaf node
                    subtree_data.extend(local_node.data)
                else:
                    for c in local_node.children[::-1]:
                        if c is not None:
                            local_stack.append(c)
                            
            # Detach all the children and get their data
            cur_node.children = []
            cur_node.data = subtree_data
            
            # Register this node in a dictionary for faster access
            row_pos_map[cur_r] =  list(map(lambda x: round(x, 3), cur_node.position))
            
            # Collect the prompt IDs
            for d in cur_node.data:
                csr_row_indexes.append(cur_r)
                csr_column_indexes.append(d['pid'])
                
            # Move on to the next tile / collection
            cur_r += 1

        else:
            if len(cur_node.children) == 0:
                # Leaf node => it means this leaf is before the target level
                # We need to adjust the node's position so that it has the same
                # size as leaf nodes at the target_level
                x, y = cur_node.data[0]['x'], cur_node.data[0]['y']
                xi, yi = int((x - x0) // step_size), int((y - y0) // step_size)

                # Find the bounding box of current level of this leaf node
                xi0, yi0 = x0 + xi * step_size, y0 + yi * step_size
                xi1, yi1 = xi0 + step_size, yi0 + step_size
                row_pos_map[cur_r] = list(map(lambda x: round(x, 3), [xi0, yi0, xi1, yi1]))
                
                # Collect the prompt IDs
                for d in cur_node.data:
                    csr_row_indexes.append(cur_r)
                    csr_column_indexes.append(d['pid'])
                
                # Move on to the next tile / collection
                cur_r += 1

            else:
                for c in cur_node.children[::-1]:
                    if c is not None:
                        stack.append((c))
    
    return csr_row_indexes, csr_column_indexes, row_pos_map

def wizmap_local(directory_folder_of_wizmap: str, port: Union[int, Literal[1000, 9999]]=8000):
    
    grid = "grid.json"
    data = "data.ndjson"
    url = f"http://localhost:{port}/{directory_folder_of_wizmap}"
    wiz_url = f"https://poloclub.github.io/wizmap/?dataURL={url+data}&gridURL={url+grid}"

    print(wiz_url)

    subprocess.run(["server", f"--port={port}", "--allow_cors"], check=True)

def calculate_cluster_centers(df):
    centers = df.groupby('cluster').mean()
    try:
        centers = centers.drop(index=-1)
    except: 
        pass
    centers['cluster'] = centers.index
    return np.stack((centers['x'], centers['y']), axis=1)
    
