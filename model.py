import pandas as pd
import numpy as np
from evaluation import Evaluation 
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from preprocess_datasets import preprocess_dataset
from preprocess_datasets.preprocess_dataset import embedded_text  
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
import torch
import torch 
import joblib
import nltk
import os 

device = "cuda" if torch.cuda.is_available() else "cpu"

''' uncomment this if you not downloaded '''
# nltk.download('stopwords')




''' data reading'''
df = pd.read_csv('cleaned_dataset/Phising_URL_Dataset.csv')

''' Memory Size checking '''
# Total memory usage
df_size_inGB = (df.shape[0] * df.shape[1] * 8) / (1024**3)
total_memory = df.memory_usage(deep=True).sum()
print(f"Total memory: {total_memory} bytes")
print(f"Total memory: {total_memory / 1024**2} MB")
total_memory_inGB = total_memory / (1024**3)
print(f"Total memory: {total_memory_inGB:.3f} GB")


''' target Column '''
y = df['label']


''' Extracting the object class labels and Numeric dtype labels '''    
txt_col = df.select_dtypes(include=['object']).columns.to_list()
num_col = df.select_dtypes(exclude=['object']).columns.to_list()


df[txt_col] = df[txt_col].fillna("")
df['combined_text'] = df[txt_col].agg(' '.join, axis=1)

''' Applying the Encoding on combined_text columns'''

embdded_texts = embedded_text(df['combined_text'].to_list())
embeddings_array = np.array(embdded_texts)
# Store in DataFrame as array
df['embedding'] = list(embeddings_array) 


print(f"Embedding shape for each text: {len(df['embedding'][0])}")


''' Dropping column because  i already have their Encoding in df[embedding] columns '''
columns_to_drop = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title', 'combined_text']
df = df.drop(columns=columns_to_drop)

''' Removing Target Column from features columns'''
if 'label' in num_col:
    num_col.remove('label')
''' Scaling the umeric columns  '''
x_num_col = StandardScaler().fit_transform(df[num_col])


# Check the type and shape
print(type(df['embedding'].iloc[0]))  # Should be numpy.ndarray
print(df['embedding'].iloc[0].dtype)   # Should be float32 or float64
print(df['embedding'].iloc[0].shape)   # Should be (768,) or similar

# When using for ML
embeddings2d = np.vstack(df['embedding'].values)  # Convert back to 2D array
print("Embedding2d shape (80) :" , embeddings2d)
# Shape should be (235795, 768)

# scaling
embedded_numpy_arr = np.array(embeddings2d)
X_numpy_arr = np.array(x_num_col)

feautres_col = df.select_dtypes(include=['int64', 'float64']).columns.to_list()
feautres_col.remove('label')
numerical_feautres = df[feautres_col].values



print("Embedded col Matrix Shape(84):", embedded_numpy_arr)
print("Sparx Matrix of x Shape(87) :",X_numpy_arr)
print("Changing x in sprax matrix is completed")
x_final = np.concatenate((X_numpy_arr,embedded_numpy_arr),axis=1) # change np.cocat in df.concat for avoiding dimension error 


print("X final Shape (90) : ", x_final)
print("Columns Succesfully Vetorized ")

X_train, X_test, y_train, y_test = train_test_split(x_final, y,test_size=0.2, random_state=42) # Added test_size and random_state for reproducibility


x_train = np.array(X_train)
x_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(f"x_train shape : {x_train.shape}")
print(f"x_test shape : {x_test.shape}")
print(f"y_test shape : {y_test.shape}")
print(f"y_train shape : {y_train.shape}")

logistic_regression = LogisticRegression(max_iter=1000,random_state=0,penalty='l2',show_progress_bar=True)
logistic_regression.fit(X=x_train,y=y_train)

y_pred = logistic_regression.predict(x_test)

y_pred = np.array(y_pred)


scores = Evaluation(y_pred=y_pred,y_test=y_test)

accuracy ,precision , recall, f1 ,c_matrix = scores.logistic_model_scores()


print("Accuracy  Scores : ", accuracy)
print("Precision Scores : ",precision)
print("Recall Scores : ", recall)
print("F1 Scores : ", f1)
print("Confusion Matrix Scores : ", c_matrix)


if f1 >= 0.8:
    joblib.dump(logistic_regression,"phising_model.joblib")
os.makedirs('cleaned_EmbeddedDF', exist_ok=True)
df.to_csv('cleaned_EmbeddedDF/Preprocess_Embededded_phishingDF.csv')
