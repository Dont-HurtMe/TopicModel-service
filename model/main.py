from .generate_vector import Generate_Vector
from .generate_topic import Mini_Bert
from .generate_wizmap import Generate_Wizmap

# model 
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import MeanShift
from sklearn.feature_extraction.text import CountVectorizer

# utility
import argparse
import yaml

def main(parameter):

    # input 
    with open('.input/parameters.yaml', 'r') as file:
        params = yaml.safe_load(file)

    embed_params = params.get('SentenceTransformer')
    reducer_params = params.get('UMAP')
    countVector_params = params.get('CountVectorizer')
    clustering_params = params.get('MeanShift')
    save_path = params.get('save_result_path')
    batch_size = params.get('batch_size')
    gpt_key = params.get('gpt_key')

    embed = SentenceTransformer(**embed_params)
    umap = UMAP(**reducer_params)
    cv = CountVectorizer(**countVector_params)
    ms = MeanShift(**clustering_params)

    vector_process = Generate_Vector(embedding_model=embed, 
        reduce_dimension_model=umap, 
        batch_size=batch_size, 
        verbose=verbose, 
        save_path=save_path)
    
    topic_process = Mini_Bert(cv=cv, 
        cluster=ms, 
        n_topic=10, 
        verbose=verbose, 
        gpt_token=gpt_key, 
        save_path=save_path)
    
    wizmap_process = Generate_Wizmap(save_file_path=save_path, 
        cv=cv, 
        path_model_log=save_path, 
        verbose=verbose)

    # transforms word2vec and reduction dimensions
    vector_process.input(docs)
    model_log, docs = vector_process.run()

    # find topic & summary
    topic_process.input(docs)
    topwords, model_log2, docs = topic_process.run()

    # generate wizmap
    wizmap_process.input(docs) 
    wizmap_process.run()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='[--top_n_word, --verbose]')
    parser.add_argument('--top_n_words', type=int, default=10, help="Input of top n of words in topic [Int]")
    parser.add_argument('--verbose', 
                        type=bool, 
                        help="Show progress of process", 
                        default=True, 
                        choices=[True, False])
    
    args = parser.parse_args()

    main(args)

