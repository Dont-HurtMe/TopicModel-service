from .generate_vector import *
from .generate_topic import *
from .generate_wizmap import *

import pandas as pd
import os

# type
from typing import List, Tuple, Union, Mapping, Any, Callable, Iterable, Dict
from pathlib import Path

# model
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import MeanShift
from sklearn.feature_extraction.text import CountVectorizer

class TopicModel : 

    def __init__(self,  
                 save_folder : str,
                 embedding : SentenceTransformer = None, 
                 reducer : UMAP = None, 
                 clustering : MeanShift = None,
                 countvector : CountVectorizer = None, 
                 gpt_api_key : str = None,
                 gemini_api_key : str = None,
                 batch_size : int = 126, 
                 device : Union[str, int] = 'cpu',
                 verbose : bool = True, 
                 llm_batch_size : int = 15000, 
                 n_sample_summary: int = 30, 
                 n_topic: int = 10):
        
        self.save_folder = save_folder
        os.makedirs(os.path.join(self.save_folder), exist_ok=True)

        self.device = device
        self.embedding = embedding or SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.reducer = reducer or UMAP()
        self.clustering = clustering or MeanShift(bandwidth=0.3)
        self.countvector = countvector or CountVectorizer(stop_words="english", ngram_range=(1, 1))
        self.gpt_api_key = gpt_api_key
        self.gemini_api_key = gemini_api_key
        self.batch_size = batch_size 
        self.verbose = verbose
        self.llm_batch_size = llm_batch_size
        self.n_sample_summary = n_sample_summary
        self.n_topic = n_topic

        self.vector = Generate_Vector(embedding_model=self.embedding, 
            reduce_dimension_model=self.reducer, 
            batch_size=self.batch_size,
            verbose=self.verbose, 
            save_path=self.save_folder)

        self.find_topic = Mini_Bert(cv=self.countvector, 
            cluster=self.clustering,    
            n_topic=self.n_topic, 
            verbose=self.verbose, 
            gpt_token=self.gpt_api_key, 
            gemini_token=self.gemini_api_key,
            save_path=self.save_folder, 
            n_sample_summary=self.n_sample_summary)
                    
        self.get_wizmap = Generate_Wizmap(save_file_path=self.save_folder, 
            cv=self.countvector, 
            verbose=self.verbose)

    def params_llm_set(self, 
                       model_name: str = None, 
                       max_tokens: int = 4096, 
                       temperature: float = 0.7):

        self.find_topic.params_llm_set(model_name=model_name, 
            max_tokens=max_tokens, 
            temperature=temperature)
        
    def fit_transforms(self, docs: List[Union[str, float, int, Any]]) -> dict:

        self.vector.input(docs)
        model_log, docs = self.vector.run()
        
        self.find_topic.input(docs)
        topwords, model_log2, docs = self.find_topic.run()
        model_log = {**model_log, **model_log2}

        self.get_wizmap.input(docs)
        self.get_wizmap.run()
        
        return {'data':docs, 'model':model_log, 'topword': topwords}
    
    
def local_graph(directory_folder_of_wizmap: str, port: Union[int, Literal[1000, 9999]]=8000):
    
    grid = "grid.json"
    data = "data.ndjson"
    url = f"http://localhost:{port}/{directory_folder_of_wizmap}"
    wiz_url = f"https://poloclub.github.io/wizmap/?dataURL={url+data}&gridURL={url+grid}"

    print(wiz_url)

    subprocess.run(["server", f"--port={port}", "--allow_cors"], check=True)



        









