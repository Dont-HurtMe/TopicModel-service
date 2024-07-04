import os
import json
import random
import pickle
import torch 
import time 
import pandas as pd
import numpy as np
import yaml
import openai
import warnings
import random 

from scipy.sparse import csr_matrix
from tqdm.autonotebook import tqdm

# type
from typing import List, Tuple, Union, Mapping, Any, Callable, Iterable, Dict
from pathlib import Path

# model
from sklearn.cluster import MeanShift
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import google.generativeai as genai

class Mini_Bert :

    def __init__(self,
                 cv: CountVectorizer = None,
                 cluster: MeanShift = None, 
                 n_topic: int = 10, 
                 verbose: bool = False, 
                 gpt_token: str = None,
                 gemini_token: str = None,
                 save_path: Path = None,
                 batch_size: int = 126, 
                 n_sample_summary: int = 30):
        
        self.top_n_words = n_topic 
        self.n_sample_summary = n_sample_summary
        self.verbose = verbose
        self.cv = cv or CountVectorizer 
        self.cluster = cluster or MeanShift(bandwidth=0.4)
        self.model_log = {}
        self.gpt_token = gpt_token
        self.gemini_token = gemini_token

        if self.gpt_token is not None and self.gemini_token is not None: 
            self.gpt_token = None

        self.tf_idf = find_topic(self.top_n_words, cv=self.cv)
        self.save_path = save_path
        self.batch_size = batch_size

        if self.gpt_token is not None or self.gemini_token is not None :    
            self.llm = llm_summary_cluster(gpt_api=self.gpt_token, 
                gemini_api=self.gemini_token, 
                verbose=self.verbose)
            self.check_llm = 'in_use'

        else : 
            self.llm = None
            self.check_llm = 'not_use'
            
    def input(self, data: pd.DataFrame): 
        self.data = data

    def params_llm_set (self,
                        model_name: str = None, 
                        max_tokens: int = 4096, 
                        temperature: float = 0.7):
        
        if self.check_llm == 'in_use':
            self.llm.params_set(model_name=model_name, 
                max_tokens=max_tokens, 
                temperature=temperature)

    def run(self)-> [pd.DataFrame, dict, pd.DataFrame]:

        ATTEMP = 10

        # clustering  
        
        vector = self.data[['x','y']].values
        self.cluster.fit(vector)
        vector = None

        cluster_list = self.cluster.labels_
        self.data['cluster'] = cluster_list
        
        # tfidf and llm summary per cluster
        if self.check_llm == 'in_use': 

            result = []
            cluster_list = list(self.data['cluster'].unique())

            cluster_iterator = tqdm(cluster_list, desc='Cluster ') if self.verbose else cluster_list

            for cluster in cluster_iterator: 

                df_tmp = self.data[self.data['cluster']==cluster]
                docs = df_tmp['data_process'].tolist()
                self.tf_idf.input(' '.join(docs))
                topword = self.tf_idf.run()
                n_sample = int(df_tmp.shape[0] * (self.n_sample_summary/100))
                xy = np.stack((self.data['x'].tolist(), self.data['y'].tolist()), axis=1)
                arr = df_tmp[['x','y','data_process']].values

                try : 
                    centers = self.cluster.cluster_centers_
                    centers = centers[cluster]
                    nearest_docs = search_nearest_points(arr=arr, 
                        centers=centers, 
                        sample=n_sample)
                    
                except : 
                    centers = calculate_cluster_centers(df_tmp[['x','y','cluster']])
                    nearest_docs = search_nearest_points(arr=arr, 
                        centers=centers[0], 
                        sample=n_sample)
                
                k = len(nearest_docs)
                for _ in range(ATTEMP) : 
                    summary = self.llm.response(nearest_docs)
                    try : 
                        test = summary
                        eval(test)
                        break 
                    except : 
                        nearest_docs = random.sample(nearest_docs, k)
                        length_current_data = len(nearest_docs)
                        k = k - 5
                        if k < length_current_data or k == 0 : 
                            k = length_current_data
                        else :
                            pass
                        summary = ''

                result.append({'cluster':cluster, 'topword':topword, 'summary':summary})

                cluster = None 
                topword = None 
                summary = None

            data_rs = []
            err = []

            result = pd.DataFrame(result)   

        if self.check_llm == 'not_use' : 

            result = []
            cluster_list = list(self.data['cluster'].unique())
            cluster_iterator = tqdm(cluster_list, desc='Cluster ') if self.verbose else cluster_list

            for cluster in cluster_iterator : 

                df_tmp = self.data[self.data['cluster']==cluster]
                docs = df_tmp['data_process'].tolist()
                self.tf_idf.input(' '.join(docs))
                topword = self.tf_idf.run()
                result.append({'cluster':cluster, 'topword':dict(topword[0])})

            result = pd.DataFrame(result)
        
        self.data['data_display'] = self.data['data_display'] + " Cluster" + self.data['cluster'].astype(str)
        self.model_log['clustering'] = self.cluster
        self.model_log['cv'] = self.cv 

        if self.save_path is not None : 

            os.makedirs(self.save_path, exist_ok=True)
            self.data.to_parquet(os.path.join(self.save_path, 'data.parquet'))

            with open(os.path.join(self.save_path, f'topic.pkl'), 'wb') as pickle_file:
                pickle.dump(result, pickle_file)

        return [result, self.model_log, self.data]
    
    def extract_output(self, list_result):
        data = [k for i in list_result for j in i for k in j]
        return data
    
class llm_summary_cluster :
    
    def __init__(self, 
                 gpt_api: str = None, 
                 gemini_api: str = None, 
                 verbose: bool = False):
        
        self.gpt_api = gpt_api
        if self.gpt_api is not None :
            openai.api_key = self.gpt_api
            self.model_name = "gpt-3.5-turbo"
            self.max_tokens = 4096
            self.temperature = 0.7

        self.gemini_api = gemini_api
        if self.gemini_api is not None : 
            genai.configure(api_key=self.gemini_api)
            self.gemini = genai.GenerativeModel('gemini-pro')

        if self.gpt_api is None and self.gemini_api is None: 
            warnings.warn("API-Key should be not None. Process will use TFIDF only.", Warning) 

        self.verbose = verbose
        
    def generate_prompt(self, docs):
        
#         prompt = f"""
#     please use the input to create and summarize short content in 'summary' and create short topic or header in 'header' and response in json format.
#     # input
#     {docs}
    
#     # example json output
# {{'header':'header', 'summary': 'summary', 'detail': {{"title": "title1", "description: "description1"}},{{"title": "title2", "description: "description2"}} }}"""

        prompt = f"""
    please use the input to create and summarize short content in 'summary' and create short topic or header in 'header' and response in json format.
    # input
    {docs}
    
    # example json output
{{'header':'header', 'summary': 'summary'}}"""
        return prompt
    
    def params_set (self, 
                    model_name: str = None, 
                    max_tokens: int = 4096, 
                    temperature: float = 0.7):
        
        if self.gpt_api is not None : 
            self.model_name = model_name or "gpt-3.5-turbo"
            self.max_tokens = max_tokens
            self.temperature = temperature

        if self.gemini_api is not None : 
            model_name = model_name if model_name is not None else 'gemini-pro'
            self.gemini = genai.GenerativeModel(model_name) 

    def response(self, text_list: List[str]):
        
        text_list = ' '.join(text_list)
        prompt = self.generate_prompt(text_list)

        # GPT
        if self.gpt_api is not None :

            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature)
            
            return self.extract_text(response.choices[0].message['content'].strip())
            
        # Gemini
        if self.gemini_api is not None : 

            N_ATTEMP = 10
            for _ in range(N_ATTEMP) : 
                try : 
                    llm_response = self.gemini.generate_content(prompt)
                    # return self.extract_text(llm_response.text)
                    return self.extract_text(llm_response.text)
                
                except Exception as e:
                    if self.verbose : 
                        warnings.warn(f"create summary fail\te:{e} attemp new one")
                    time.sleep(random.randint(1, 5))
            return ''
        
    def extract_text(self, txt):
        for pattern in ['json', '```', '\n']:
            txt = txt.replace(pattern, '')
        return txt 
    
class find_topic :
    
    def __init__(self, 
                 n_topic: int = 10, 
                 cv: CountVectorizer = None):
        
        self.n_topic = n_topic 
        self.cv = cv or CountVectorizer(stop_words="english", ngram_range=(1, 1))
        self.t_tf_idf_model = TfidfTransformer()

    def input(self, documents: str):
        self.documents = documents
        
    def top_n_idx_sparse(self, 
                         matrix: csr_matrix, 
                         n: int) -> np.ndarray:
            
        """Return indices of top n values in each row of a sparse matrix"""
        indices = []
        for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
            n_row_pick = min(n, ri - le)
            values = matrix.indices[
                le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]
            ]
            values = [
                values[index] if len(values) >= index + 1 else None for index in range(n)
            ]
            indices.append(values)
        
        return np.array(indices)

    def top_n_values_sparse(self, 
                            matrix: csr_matrix, 
                            indices: np.ndarray) -> np.ndarray:
        
        """Return the top n values for each row in a sparse matrix"""
        top_values = []
        for row in range(indices.shape[0]):
            scores = np.array(
                [matrix[row, c] if c is not None else 0 for c in indices[row, :]]
            )
            top_values.append(scores)
            
        return np.array(top_values)

    def count_idf(self):
        
        count_mat = self.cv.fit_transform([self.documents])
        t_tf_idf = self.t_tf_idf_model.fit_transform(count_mat)
        ngrams = self.cv.get_feature_names_out()

        return [count_mat, t_tf_idf, ngrams]

    def run(self, ):

        count_mat, t_tf_idf, ngrams = self.count_idf()
        
        # Get indices of top-k words with highest TF-IDF scores
        indices = self.top_n_idx_sparse(t_tf_idf, self.n_topic)
        scores = self.top_n_values_sparse(t_tf_idf, indices)

        tile_topics = []
        
        for r in range(t_tf_idf.shape[0]):
            word_scores = [
                (ngrams[indices[r][i]], round(scores[r][i], 4))
                if indices[r][i] is not None and scores[r][i] > 0
                else ("", 0.00001)
                for i in range(self.n_topic)
            ]
        
            tile_topics.append(word_scores)

        return tile_topics
    
def calculate_cluster_centers(df: pd.DataFrame):

    centers = df.groupby('cluster').mean()
    try:
        centers = centers.drop(index=-1)
    except: 
        pass
    centers['cluster'] = centers.index
    centers = np.stack((centers['x'].tolist(), centers['y'].tolist()), axis=1)

    return centers
    
def search_nearest_points(arr: np.ndarray, 
                          centers: np.ndarray, 
                          sample: int = 50) -> list:

    coordinates = arr[:, :2].astype(float)
    data_process = arr[:, 2]

    x1, y1 = centers
    distances = np.sqrt((coordinates[:, 0] - x1)**2 + (coordinates[:, 1] - y1)**2)
    nearest_indices = np.argsort(distances)[:sample]
    nearest_points = data_process[nearest_indices]
    
    return nearest_points.tolist()
    
