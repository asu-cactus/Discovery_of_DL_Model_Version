import numpy as np
import pandas as pd 
import sys
import utils
from tqdm.notebook import tqdm
from minHash import MinHash
from minHashLSH import MinHashLSH
from minHashStorage import LSHStorage

def mem_size(object):
    """Return object size in kb

    """
    return sys.getsizeof(object) / 1024

def mem_size_dict(d):
    """Return the size of a dict object in kb (list as dict value)
    """
    size = sys.getsizeof(d)
    for key in d.keys():
        size += sys.getsizeof(d[key])
    return size / 1024

def mem_size_dict_minhash(d):
    """Return the size of a dict obj in kb (MinHash as dict value)
    """
    size = mem_size(d)
    for key in d.keys():
        size += mem_size_minhash(d[key])
    return size

def mem_size_minhash(minhash):
    """Return the size of a MinHash obj in kb
    """
    size_permu = mem_size(minhash.num_permu)
    size_coef = mem_size(minhash.permutations_coef)
    size_matrix = mem_size(minhash.hash_matrix)
    size_total = size_permu + size_coef + size_matrix
    
    return size_total

def mem_size_minhashlsh(lsh):
    """Return the size of a MinHashLSH obj in kb
    """
    size_hashrange = mem_size(lsh.hash_ranges)
    size_hashtable = mem_size_dict(lsh.hash_table)
    size_keys = mem_size(lsh.keys)
    size_total = size_hashrange + size_hashtable + size_keys
    return size_total

def mem_size_lshstorage(storage):
    """Return the size of a LSHStorage obj in kb
    """
    size_tables = mem_size(storage.tables)
    size_cols = mem_size_dict(storage.cols)
    size_minHashs = mem_size_dict_minhash(storage.minHashs)
    size_lsh = mem_size_minhashlsh(storage.minHashLSH)
    size_storage = size_tables + size_cols + size_minHashs +size_lsh
    return size_storage

def actual_jaccard(data1, data2):
    """A util function to get the real 'jaccard similarity'

    Args:
        data1: array-like of shape
        data2: array-like of shape
    
    Returns:
        float: the real jaccard similarity
    """

    s1 = set(data1)
    s2 = set(data2)
    actual_jaccard = float(len(s1.intersection(s2)) / len(s1.union(s2)))

    return actual_jaccard

def confusion_matrix(actual_dict, expect_dict, evaluate_keys, all_keys):
    t_p, t_n, f_p, f_n = 0, 0, 0, 0

    for e_key in evaluate_keys:
        # Evaluate {e_key: [key1, key2]}
        for key in all_keys:
            # True Positive
            if key in expect_dict[e_key] and key in actual_dict[e_key]:
                t_p += 1
                continue
            # True Negative
            if key not in expect_dict[e_key] and key not in actual_dict[e_key]:
                t_n += 1
                continue
            # False Positive
            if key not in expect_dict[e_key] and key in actual_dict[e_key]:
                f_p += 1
                continue
            # Flase Negative
            if key in expect_dict[e_key] and key not in actual_dict[e_key]:
                f_n += 1
                continue
                
    confusion_matrix = np.array([[t_p, f_n], [f_p, t_n]])
    cm_df = pd.DataFrame(confusion_matrix, index = ['Positive', 'Negative'], columns = ['Positive', 'Negative'])

    return cm_df

def performance_measure(confusion_matrix):
    """Return [accuracy, precision, recall, f1 score] based on given confusion matrix

    Args:
        confusion_matrix (2d array): given based on the following format:
            [[true positive, false negative],
             [false positive, true negative]]

    """
    tp = confusion_matrix[0][0]
    fn = confusion_matrix[0][1]
    fp = confusion_matrix[1][0]
    tn = confusion_matrix[1][1]
    accuracy = (tp+tn) / (tp+fn+fp+tn)
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1 = 2*precision*recall / (precision + recall)
    return [accuracy, precision, recall, f1]
    
def acc_num_permu(df, permu_range=range(1, 2*16, 2), example_size=1, 
                  sample_frac=0.2, loop=20, random_state=0,
                  len_col=4):
    #TODO Add comments

    # list_num_permu = range(1, 2*16, 2)
#     example_size = 1
#     frac = 0.2
#     len_col = 4

    # loop = 20

    sample = df.sample(frac=sample_frac, random_state=random_state).iloc[:, :len_col]

    list_hit = []

    for num_permu in tqdm(permu_range):
        num_hit = 0
        for i in tqdm(range(loop), leave=False):
            lsh = MinHashLSH(num_permu=num_permu, threshold=0, duplicate_key=True)
            utils.insert_df_row_lsh(lsh, sample, 'df', num_permu)

            user_example = df.sample(n=example_size).iloc[:, :len_col]
            result = utils.lsh_hit_df_row(lsh, user_example, num_permu)
            if 'df' in result:
                num_hit += 1

        list_hit.append(num_hit/loop)
    
    return list_hit

def acc_example_size(df, num_permu=5, example_range=range(1, 16, 1), 
                     sample_frac=0.2, loop=20, random_state=0,
                     len_col=4):
    # list_example_size = range(1, 16, 1)
#     frac = 0.2
    # len_col = 4
#     loop = 20
    sample = df.sample(frac=sample_frac, random_state=random_state).iloc[:, :len_col]

    list_hit = []

    for example_size in tqdm(example_range):
        num_hit = 0
        for i in tqdm(range(loop), leave=False):
            lsh = MinHashLSH(num_permu=num_permu, threshold=0, duplicate_key=True)
            utils.insert_df_row_lsh(lsh, sample, 'df', num_permu)

            user_example = df.sample(n=example_size).iloc[:, :len_col]
            result = utils.lsh_hit_df_row(lsh, user_example, num_permu)
            if 'df' in result:
                num_hit += 1

        list_hit.append(num_hit/loop)
    return list_hit

def acc_sample_size(df, num_permu=5, example_size=1, 
                    sample_range=np.arange(0.05, 1.05, step = 0.05), 
                    loop=20, random_state=0, len_col=4):
    # list_sample_frac = np.arange(0.05, 1.05, step = 0.05)
    
    list_hit = []
    # len_col = 4
#     loop = 20
#     example_size = 1
        
    for frac in tqdm(sample_range):
        
        sample = df.sample(frac=frac, random_state=random_state).iloc[:, :len_col]
        
        num_hit = 0
        
        for i in tqdm(range(loop), leave=False):
            lsh = MinHashLSH(num_permu=num_permu, threshold=0, duplicate_key=True)
            utils.insert_df_row_lsh(lsh, sample, 'df', num_permu)
            
            user_example = df.sample(n=example_size).iloc[:, :len_col]
            result = utils.lsh_hit_df_row(lsh, user_example, num_permu)
            if 'df' in result:
                num_hit += 1
        
        list_hit.append(num_hit/loop)    
    return list_hit