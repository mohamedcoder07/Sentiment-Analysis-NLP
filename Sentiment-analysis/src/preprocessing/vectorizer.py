from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
from transformers import AutoTokenizer, AutoModel



BASE_DIR = Path.cwd()
print(BASE_DIR)
tokenizer_path = BASE_DIR / "saved_models" / "tokenizer"
roberta_path = BASE_DIR / "saved_models" / "roberta_model"


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")    


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask




class WordVectorizer(BaseEstimator):

    def __init__(self, method = "tfidf"):
        
        self.method = method        

        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(stop_words = "english", min_df=5)
        
        elif self.method == "transformers":
            print("This analysis is done using RoBERTa model !")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            self.model = AutoModel.from_pretrained(roberta_path).to(device)



    def fit(self, texts):        
        if self.method == "tfidf":
            self.vectorizer.fit(texts)
            return self
        elif self.method == "transformers":
            return self
    

    def transform(self, texts):        
        if self.method == "tfidf":
            return self.vectorizer.transform(texts)
        elif self.method == "transformers":
            return self.get_embeddings(texts, batch_size= 16)


    def fit_transform(self, texts):        
        if self.method == "tfidf":
            return self.vectorizer.fit_transform(texts)    
        elif self.method == "transformers":
            return self.tranform(texts)
        
    def get_embeddings(self, texts, batch_size = 16):

        embeddings = []
        
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()  
        elif isinstance(texts, str):
            texts = [texts]
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = texts[i:i+batch_size]
                encoded = self.tokenizer(batch, padding = True, truncation = True, max_length = 128, return_tensors = "pt")
                attention_mask = encoded['attention_mask']
                outputs = self.model(**encoded)            
                batch_emb = mean_pooling(outputs.last_hidden_state, attention_mask).cpu().numpy()
                embeddings.append(batch_emb)
        return np.vstack(embeddings)    

