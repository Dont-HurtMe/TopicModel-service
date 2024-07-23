# **dsm_wizservice**
- demo service topic model
- backup code

## **Example use:**

### **Install** 
```
pip install git+https://github.com/Dont-HurtMe/TopicModel-service.git
```
### **Import** 
```python
from dsm_wizservices import TopicModel
```

### **Input data**
```python
docs = ['sentence_1', 'sentence_2', 'sentence_3', ... 'sentence_n']
```

### **Defind Parameters**
- **Quick start**
```python
tp_model = TopicModel(save_folder='./output_test/')
```

- **Example Defind Parameters**
```python
from sentence_transformers import SentenceTransformer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import MeanShift
from umap import UMAP

# API_KEYS
gemini_key = <gemini_API_KEY>

# embedding model
s_bert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# reduction dimensions model
umap = UMAP(n_components=2, n_neighbors=60, min_dist=0.2)

# BOW (bag of word)
countvec = CountVectorizer(stop_words="english", ngram_range=(1, 1))

# Clustering model
cluster_model = MeanShift(bandwidth=0.7)
```
```python
tp_model = TopicModel(save_folder='output_test/', # path directory to save wizmap data [grid.json, data.ndjson] 
    # gpt_api_key=api_key, # if apikey is gpt api_key 
    n_sample_summary=20, # sample of sammary in each cluster
    gemini_api_key=gemini_key, 
    embedding=s_bert_model, 
    reducer=umap, 
    clustering=cluster_model, 
    countvector=countvec, 
    batch_size=126, # use when embedding words to vector
    device='cpu', # device for s_bert or embedding model
    verbose=True)

tp_model.params_llm_set(model_name='gemini-pro') # example set parameters of llm
```

### **Get wizmap and output**
```python
result = tp_model.fit_transforms(docs)
```

### **Details of output**
After generating these has two JSON files (one with grid.json and one with data.ndjson). You can store them in somewhere network.

```python
# result type is dictionary 2 keys
# key1 annotation of each points after generate.
dataframe = result['data']
# key2 dataframe of summary and topword in each cluster this can help to analysis.
topic_dataframe = result['topword']
```

### **Run Wizmap on local**
```python
from dsm_wizservices import plot_local

plot_local(directory_folder_of_wizmap='output_test/wizmapJson/')
```


### Jupyter
#### step by step & one shot 
- <a href="/notebook/_00-example-use.ipynb"> _00-example-use.ipynb </a>


