"""
Class implementing the recommendation logic
"""
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from dataset_creation import create_dataset

CURRENT_PATH = Path(__file__)
DATA_PATH = CURRENT_PATH.parent.parent/'data'
TABULAR_PATH = DATA_PATH/'tabular_data.csv'
METADATA_PATH = DATA_PATH/'metadata.json'
BASE_DATA_PATH = DATA_PATH/'base_data.csv'

def calculate_numeric_diff(df:pd.DataFrame,row_num:int,col:str):
    """Calculate the exponential difference for a column of the data and its row

    Parameters
    ----------
    df : pd.DataFrame
        dataframe of all of the values
    row_num : int
        row to calculate the difference with all of the other
        columns
    col : str
        name of the numeric column to use

    Returns
    -------
    pd.DataFrame
        dataframe of the resulting differences
    """
    exp_diff = df[col].apply(lambda x:(np.exp((x-df.loc[row_num,col])/df[col].max())))
    return exp_diff

def compare_with_binarized_columns(df:pd.DataFrame,row_num:int,metadata:Dict[str,str]):
    """binarize the categorical columns for comparison using the metadata

    Parameters
    ----------
    df : pd.DataFrame
        dataframe of the categorical columns (or the entire data)
    row_num : int
        row to compare with
    metadata : Dict[str,str]
        The datatype of each corresponding column used when creating
        the data

    Returns
    -------
    pd.DataFrame
        dataframe with the binary matches
    """
    total_binarized = []
    for col,dtype in metadata.items():
        if dtype=='categorical':
            column = df[col]
            col_match = (column==df.loc[row_num,col]).astype(int).rename(f'{col}_match')
            total_binarized.append(col_match)

    total_match = pd.concat(total_binarized,axis = 1)
    return total_match

def compute_cosine_similarity(vec_rec,matrix):
    """Compute cosine similarity between a vector rows of a matrix

    Parameters
    ----------
    vec_rec : np.ndarray
    matrix : np.ndarray

    Returns
    -------
    np.ndarray
    """
    cos_df = cosine_similarity([vec_rec],matrix)
    cos_df = np.squeeze(cos_df)
    return cos_df

class Recommender:
    """Class implementing recommendation logic using the stored data
    """
    def __init__(self,weights = None,original_data_path = BASE_DATA_PATH):
        """initialize the class with the given weights for each component

        Parameters
        ----------
        weights : np.ndarray, optional
            weights used for each column, by default None
        original_data_path : str, optional
            path to the original data, by default BASE_DATA_PATH
        """
        #keep the columsn in lists to preserve order later
        self.image_columns = []
        self.text_columns = []
        self.categorical_columns = []
        self.numeric_columns = []
        #load the original data
        self.original_data = pd.read_csv(original_data_path)

        #loading the metadata
        with open(METADATA_PATH,'r') as metadata:
            self.metadata = json.load(metadata)

        #load the stored data
        for column, dtype in self.metadata.items():
            if dtype in ['image','text']:
                with open(DATA_PATH/(column+'_array.npy'),'rb') as f:
                    data_array = np.load(f,allow_pickle=True)
                    setattr(self,column,data_array)

                if dtype=='text':
                    self.text_columns.append(column)
                else:
                    self.image_columns.append(column)

            elif dtype=='numeric':
                self.numeric_columns.append(column)
            else:
                self.categorical_columns.append(column)

        self.tabular = pd.read_csv(TABULAR_PATH)

        if weights:
            self.weights = weights
        else:
            #initialize uniform weights
            n = len(self.metadata.keys())
            self.weights = np.ones(n)/n

    def predict(self,row_number,top_k = 5):
        """return the most similar items to the given row in the existing data

        Parameters
        ----------
        row_number : int
        top_k : int
            the number of rows to retrieve

        Returns
        -------
        pd.DataFrmae
            the most similar items
        """
        similarity_scores = []

        #add the numeric differences
        for numeric_col in self.numeric_columns:
            similarity_score = calculate_numeric_diff(
                self.tabular,
                row_number,
                numeric_col).to_numpy()
            similarity_scores.append(similarity_score)

        #add matching for categorical columns
        binary_results = np.hsplit(
            compare_with_binarized_columns(
                self.tabular,
                row_number,
                self.metadata
                ).to_numpy(),
                len(self.categorical_columns
                )
            )
        binary_results = [np.squeeze(result) for result in binary_results]
        similarity_scores.extend(binary_results)

        #computing similarity for the image and text columns
        for array_column in self.image_columns + self.text_columns:
            array = getattr(self,array_column)
            similarity_score = compute_cosine_similarity(array[row_number,:],array)
            similarity_scores.append(similarity_score)

        #handling of duplicated pictures
        starting_index = len(self.numeric_columns)+len(self.categorical_columns)
        for index,image_column in enumerate(self.image_columns):
            dataset = self.original_data.copy(deep = True)
            data_ = dataset[~dataset[image_column].duplicated()][image_column]
            image_to_cosine_sim_dict = {
                image_name:img_array for image_name,img_array in
                zip(data_.to_numpy(),similarity_scores[starting_index+index])
                }
            similarity_scores[starting_index+index] = dataset[image_column].map(image_to_cosine_sim_dict).to_numpy()

        #reshape the arrays and concatenate them as columns
        similarity_scores = [
            np.expand_dims(arr,axis = 1).reshape(-1,1).astype(np.float32) 
            for arr in similarity_scores
            ]
        total_similarities = np.concatenate(similarity_scores,axis = 1)

        #rescale the similarity scores
        scaler = MinMaxScaler()
        similarities_array = pd.DataFrame(scaler.fit_transform(total_similarities))
        similarities_array['weighted_sum'] = np.matmul(
            similarities_array.to_numpy().astype(float),
            self.weights
            )

        #retrieve the top 5 rows
        topk = similarities_array["weighted_sum"].sort_values(ascending = False).index[:top_k]
        top_k_data = self.original_data.filter(items = topk.values, axis=0)
        return top_k_data

    def problem_solver(self):
        """Function for printing the problem definition
        """
        result = '''A class implementing the recommendation logic for a stored data.
        Use the predict method to get the predictions for a given item from the stored data.'''

        return result

    def __str__(self):
        return f'Recommender with weights {self.weights}'

if __name__=='__main__':
    rec = Recommender()
    print(rec.predict(0,top_k=5))
