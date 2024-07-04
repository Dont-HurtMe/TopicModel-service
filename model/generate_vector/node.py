import os
import random
import pickle
import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
from scipy.sparse import csr_matrix

# type
from typing import List, Tuple, Union, Mapping, Any, Callable, Iterable, Dict
from pathlib import Path

# model
from sentence_transformers import SentenceTransformer
from umap import UMAP

# find topic 
# from scipy.sparse import csr_matrix
# from sklearn.metrics.pairwise import cosine_similarity

# data_dir = "../../data_store"

class Generate_Vector:
    
    def __init__(self, 
                 embedding_model: SentenceTransformer = None, 
                 reduce_dimension_model: UMAP = None, 
                 batch_size: int = 64, 
                 verbose: bool = False, 
                 save_path: Path = None):
        
        self.verbose = verbose
        self.embedding_model = embedding_model or SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        self.reduce_dimension_model = reduce_dimension_model or UMAP(random_state=random.randint(0, 2**32 - 1),verbose=self.verbose)
        self.batch_size = batch_size
        self.save_path = save_path

    def input(self, data: List[str]):
        self.data = data
        
    def word_embedding(self, docs: List[Union[float, int, str]]) -> np.array:
        
        vector = self.embedding_model.encode(docs, 
                    show_progress_bar=self.verbose, 
                    batch_size=self.batch_size)
        
        return vector
    
    def reduce_dimension(self, vector: np.array) -> UMAP:
        reduced_vector = self.reduce_dimension_model.fit(vector)
        return reduced_vector
        
    def run(self) -> [Dict[str, Any], pd.DataFrame]:

        # load-data

        if self.verbose : 
            print("embedding -> reduction dimension ...")

        vector = self.word_embedding(self.data)
        self.reduce_dimension(vector)
        
        self.data = pd.DataFrame(self.data, columns=['data_process'])
        self.data['vector'] = [i for i in vector]
        self.data['x'] = self.reduce_dimension_model.embedding_[:,0]
        self.data['y'] = self.reduce_dimension_model.embedding_[:,1]
        self.data['data_display'] = "X: " + round(self.data['x'], 4).astype(str) + " Y: " + round(self.data['y'], 4).astype(str) + " Content: " + self.data['data_process']

        model_log_dict = {'sentence_transform': {'model':self.embedding_model, 'vector':vector}, 
                          'umap': self.reduce_dimension_model}
        
        if self.save_path is not None : 

            os.makedirs(self.save_path, exist_ok=True)
            self.data.to_parquet(os.path.join(self.save_path, 'data.parquet'))

            with open(os.path.join(self.save_path, f'model.pkl'), 'wb') as pickle_file:
                pickle.dump(model_log_dict, pickle_file)

        return [model_log_dict, self.data]