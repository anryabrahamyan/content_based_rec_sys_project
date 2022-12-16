"""
Script for creating the data to be used by the recommender class
"""
import json
import os
import urllib
from pathlib import Path
from shutil import rmtree
from typing import Dict

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50
from transformers import RobertaTokenizer, TFRobertaModel

CURRENT_PATH = Path(__file__)
DATA_PATH = CURRENT_PATH.parent.parent/'data'

MODEL_NAME = "roberta-base"
MAX_LEN = 200 # maximum length of the textual values processed by the NLP model

#defining necessary functions for data processing
def build_model():
    """Create the model for vectorizing texts

    Returns
    -------
    tf.Model
        model for predictions on the texts
    """
    input_word_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_type_ids')

    # Import RoBERTa model from HuggingFace
    roberta_model = TFRobertaModel.from_pretrained(MODEL_NAME)
    x = roberta_model(input_word_ids, attention_mask=input_mask, token_type_ids=input_type_ids)

    # Huggingface transformers have multiple outputs, embeddings are the first one,
    # so let's slice out the first position
    x = x[0]

    model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    return model

def roberta_encode(texts, tokenizer):
    """function used for encoding the texts given a tokenizer

    Parameters
    ----------
    texts : pd.Series[str]
        columns of texts to be encoded
    tokenizer : transformers tokenizer
        Tokenizer for preparing the inputs for the appropriate model
        as input

    Returns
    -------
    Dict[str,np.ndarray]
        Encodings returned by the tokenizer
    """
    ct = len(texts)
    input_ids = np.ones((ct, MAX_LEN), dtype='int32')
    attention_mask = np.zeros((ct, MAX_LEN), dtype='int32')
    token_type_ids = np.zeros((ct, MAX_LEN), dtype='int32')  # Not used in text classification

    for k, text in enumerate(texts):
        # Tokenize
        tok_text = tokenizer.tokenize(text)

        # Truncate and convert tokens to numerical IDs
        enc_text = tokenizer.convert_tokens_to_ids(tok_text[:(MAX_LEN - 2)])

        input_length = len(enc_text) + 2
        input_length = input_length if input_length < MAX_LEN else MAX_LEN

        # Add tokens [CLS] and [SEP] at the beginning and the end
        input_ids[k, :input_length] = np.asarray([0] + enc_text + [2], dtype='int32')

        # Set to 1s in the attention input
        attention_mask[k, :input_length] = 1

    return {
        'input_word_ids': input_ids,
        'input_mask': attention_mask,
        'input_type_ids': token_type_ids
    }

def vectorize_column(df_col:pd.DataFrame,nlp_model):
    """function used for vectorizing columns of the data

    Parameters
    ----------
    df_col : pd.DataFrame
        Textual dataframe column to vectorize
    nlp_model : Transformer model class
        Model which should vectorize the texts

    Returns
    -------
    np.ndarray
        The text columns in vectorized form
    """
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    df_col.fillna('',inplace=True)

    encoded_df = roberta_encode(df_col,tokenizer)

    encoded_matrix = nlp_model.predict(encoded_df)
    output_reduced = np.sum(encoded_matrix,axis = 2)

    return output_reduced

def download_and_vectorize_images(df_col:pd.Series)->int:
    """Downloads the images from the given links into the data directory

    Parameters
    ----------
    df_col : pd.Series
        column of image links

    Returns
    -------
    success : int
        wether the process failed or not
    """
    #(re)create the appropriate folder
    #take each url from the column and download the image from each
    images_folder_path = DATA_PATH/(df_col.name+'_images')
    if os.path.isdir(images_folder_path):
        rmtree(images_folder_path)
    else:
        os.mkdir(images_folder_path)

    def download_image(link):
        response = requests.get(link,timeout=20)
        img_data = response.content
        url_parts = urllib.parse.urlparse(link)
        path_parts = url_parts[2].rpartition('/')
        imgSaveDir = os.path.join(images_folder_path, path_parts[2])
        with open(imgSaveDir, "wb") as handler:
            handler.write(img_data)

    try:
        df_col.apply(download_image)
        images_array = vectorize_images_in_directory(
            images_dir = images_folder_path,
            images_df=df_col
            )
        #save the data for later use
        with open(DATA_PATH / f'{df_col.name}_array.npy', 'wb') as f:
            np.save(f, images_array)
        success = 1
    except:
        success = 0

    return success

def vectorize_save_texts(df_col,model):
    """Vectorizes the given column and saves the result in the data directory

    Parameters
    ----------
    df_col : pd.DataFrame
        column to encode
    model : _type_
        _description_

    Returns
    -------
    success : int
        wether the process failed or not
    """
    try:
        vectorized_texts = vectorize_column(df_col=df_col,nlp_model=model)

        with open(DATA_PATH /f'{df_col.name}_array.npy', 'wb') as f:
            np.save(f, vectorized_texts)
        success = 1
    except:
        success = 0
    return success

def vectorize_images_in_directory(images_dir,images_df):
    """vectorizes the images in teh given directory

    Parameters
    ----------
    images_dir : str,Path
        the directory where the images are stored
    images_df : pd.DataFrame
        the dataframe containing the image links

    Returns
    -------
    np.ndarray
        vectorizes array of the stored images in the directory
    """

    lst = []
    names = []
    for image in os.listdir(images_dir):
        i = Image.open(images_dir / image)
        i = np.array(i)
        lst.append(i)
        names.append(image)

    name_image_mapping = dict()
    image_names_duplicated = images_df.apply(lambda x:x.split('/')[-1])

    lst_= [tf.image.resize(img,size=(224,224)) for img in lst]
    model = ResNet50(include_top=True,input_shape=(224, 224, 3))
    os.chdir(DATA_PATH)

    ar = np.empty((1,1000))
    for i in range(len(lst_)):
        m = model(tf.expand_dims(lst_[i],0))
        ar = np.concatenate([ar,m])
        name_image_mapping[image_names_duplicated.to_list()[i]] = m.numpy()

    name_image_mapping['NaN'] = np.nan
    prep = images_df\
        .apply(lambda x:str(x).split('/')[-1])\
        .map(name_image_mapping,na_action = 'ignore')

    prep = prep[~prep.isna()]
    vectorized_images_array = np.squeeze(np.stack(prep.to_list(),axis = 1))

    return vectorized_images_array

def create_dataset(total_data:Dict[str,str],df:pd.DataFrame)->None:
    """Function for creating the data needed for recommendation
    and product comparison

    Parameters
    ----------
    total_data : Dict[str,str]
        dictionary of colnames and their datatypes from text,image,numeric,categorical
        eg. {"col1":'numeric',"col2":'image'}
    df : pd.DataFrmae
        dataframe to be used for the dataset creation
        note: images should be links
    """
    #create the folder for storing the data for faster inference
    if os.path.isdir(DATA_PATH):
        rmtree(DATA_PATH)
        print('recreating the data folder')
    os.mkdir(DATA_PATH)

    #dataframe for storing the tabular data (numeric,categorical)
    table_data = pd.DataFrame()

    model = build_model()

    for column,dtype in total_data.items():
        col = df[column]
        if dtype =='image':
            success = download_and_vectorize_images(col)
        elif dtype == 'text':
            success = vectorize_save_texts(col,model=model)
        elif dtype == 'numeric':
            table_data = pd.concat([table_data,col.astype(np.float64)],axis = 1)
            success = 1
        elif dtype == 'categorical':
            table_data = pd.concat([table_data,col.astype("category")],axis = 1)
            success = 1
        #if any of the processes return success of 0, an error is raised
        if not success:
            raise RuntimeError(f'could not process {column} of type {dtype}')

    table_data.to_csv(DATA_PATH/'tabular_data.csv',index = False)

    #save the metadata for later use in the recommender class
    with open(DATA_PATH/'metadata.json','w') as metadata:
        json.dump(total_data,metadata)

if __name__=='__main__':
    #sample data for experimentation
    sample_df = pd.read_csv(DATA_PATH.parent/'test_dataset.csv').iloc[:100,:]
    #preprocessing to make the price column numeric
    sample_df['price'] = sample_df['price'].apply(lambda x: x.replace(',','')).astype(np.float64)
    total_data = {
        'store':'categorical',
        'price':'numeric',
        'item description':'text',
        'image url':'image',
        'category':'categorical'
    }
    create_dataset(total_data=total_data,df = sample_df)
    #the original dataset should also be saved in the data folder as base_data.csv
    sample_df.to_csv(DATA_PATH/'base_data.csv',index = False)
