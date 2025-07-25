import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from nltk import WordNetLemmatizer
import nltk
import re
from nltk.corpus import stopwords
import numpy as np
from transformers import BertTokenizer, BertModel
import random
import torch


random_seed = 42 
random.seed(random_seed)

torch.manual_seed(random_seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)



device = "cuda" if torch.cuda.is_available() else "cpu"
sen_model = SentenceTransformer("all-MiniLM-L6-v2",device=device)



project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)






def Word_embeddings_AndTokenize(text):
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    wembeddings_model = BertModel.from_pretrained('bert-base-uncased')
    
    row_loaded = 0
    chunks = 20000
    chunk_list = []
    for start in range(0,len(text),chunks):
        end = min(start+chunks,len(text))
        chunk_text = text[start:end]
        
        
        bert_encoding = bert_tokenizer.batch_encode_plus( [chunk_text],# List of input texts
        padding=True,              # Pad to the maximum sequence length
        truncation=True,           # Truncate to the maximum sequence length if necessary
        return_tensors='pt',      # Return PyTorch tensors
        add_special_tokens=True    # Add special tokens CLS and SEP
        )
        input_ids = bert_encoding['input_ids']
        attention_mask = bert_encoding['attention_mask'] 
        with torch.no_grad():
            outputs = wembeddings_model(input_ids,attention_mask=attention_mask)
            word_embeddings = outputs.last_hidden_state  # This contains the embeddings
        row_loaded += chunks
        print(f'Row Encodings Completed  of :{row_loaded} rows')
        chunk_list.append(word_embeddings)
        
    return chunk_list






def preprocess_data(df,cleaning_text):
    for col in df.columns:
        if df[col].dtype == 'object':
            if col == 'URL':
                df[col] = df[col].apply(lambda x: cleaning_text(x, is_url=True))
            else:
                df[col] = df[col].apply(cleaning_text)
            print(f"Cleaned column: {col}")
            print(df[col].head()) # Print head instead of entire column
    os.makedirs('cleaned_dataset', exist_ok=True)
    df.to_csv('cleaned_dataset/Phising_URL_Dataset.csv',index=False)
    

def embedded_text(dataset):
    chunk_size = 20000
    lst_enc_text = []
    row = 0
    for start_batch in range(0,len(dataset),chunk_size):
        batch_end = min(start_batch+chunk_size, len(dataset))
        
        batch_data = dataset[start_batch:batch_end] 
        
        enc_text = sen_model.encode_document(batch_data,convert_to_tensor=True,batch_size=100,show_progress_bar=True,device=device)
        
        embeddings = enc_text.cpu().numpy() 
        lst_enc_text.append(embeddings)
        row += len(enc_text)
        
        print(f"Encoded row Done: {row}")
    final_embeddings = np.concatenate(lst_enc_text, axis=0)
    return final_embeddings.tolist()


def preprocessed_text(text):
    text = str(text)
    lowercase_word = [word.lower() for word in text.split()]
    removed_punctuation = re.sub(r"[^a-zA-Z0-9\s]", '', str(lowercase_word))
    removed_punctuation2 = re.sub(r"http\S+", "", removed_punctuation)  # remove links
    removed_punctuation3 = re.sub(r"[^a-zA-Z]", " ", removed_punctuation2)  # keep only letters
    tokens = nltk.word_tokenize(removed_punctuation3, language="english")
    stopword = set(stopwords.words("english"))
    tokens = [word for word in tokens  if word not in stopword ]
    lemitizer = WordNetLemmatizer()
    token_lemitize = [lemitizer.lemmatize(words) for words in tokens]
    return ' '.join(token_lemitize)






