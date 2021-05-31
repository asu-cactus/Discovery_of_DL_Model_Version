import scipy.stats
import numpy as np
import time
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer


def normalization(data):
    """Normalize the data

        Args:
            data (array)
        
        Returns:
            array: normalized data
    """

    range = np.max(data) - np.min(data)

    return (data - np.min(data)) / range

def data_to_center(data):
    """Compute the center of a datasets

    Normalize the data first, then compute the average value for each attribute
    as its corresponding center

    Args:
        data (array)

    """

    data = np.average(normalization(data), axis=0)
    return data

def l2_distance(data1, data2):
    """Compute l2 distance between the given data

    Args:
        data1 (1d-array)
        data2 (1d-array)
    """

    if len(data1) != len(data2):
        raise ValueError('The size of given two data should be equal. '\
                         'One is {}, other is {}'.format(len(data1), len(data2)))

    return np.linalg.norm(data1 - data2)

def squared_hellinger_distance(prob1, prob2):
    """Compute the squared hellinger distance

    Args:
        prob1 (array): a discrete probability distribution
        prob2 (array): another discrete probability distribution
    
    Returns:
        float: squared hellinger distance
    """
    return 0.5*np.linalg.norm(np.sqrt(prob1) - np.sqrt(prob2))**2

def jensen_shannon_divergence(prob1, prob2):
    """Compute the JS - Divergence between the given probability distributions
    
    https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    
    Args:
        prob1 (array): a discrete probability distribution
        prob2 (array): another discrete probability distribution
    
    Returns:
        float: calculated JS - Divergence
    
    """

    if (len(prob1) != len(prob2)):
        raise ValueError('The sizes of two distributions should be qual.')

    m = (prob1 + prob2) / 2
    jsd = 0.5*scipy.stats.entropy(prob1, m) + 0.5*scipy.stats.entropy(prob2, m)
    
    if np.isinf(jsd):
        # the JS Divergence will be inf if there is no term in any one distribution.
        return 0
    
    return jsd


def discrete_distribution(data, range=[[0,1]], bins=100):
    """Compute a 1d discrete probability distribution

    Use np.histogramdd to compute the distribution of a flatten data with given 
    range and bins

    Args:
        data (array)
        range (list): a range of probability distribution to generate,
            default=[[0,1]]
        bings (int): number of histogram bins, default=100
    
    Return:
        array: computed 1d probability distribution

    """
    
    data = data.flatten()
    H, edges = np.histogramdd(data, range=range, bins=bins)
    
    return H / np.sum(H)


def data_to_probability(data, bins=100):
    """Compute discrete probability for the data

    Applying the normalization to the data first, then  using 
    discrete_distribution to compute the probability

    Args:
        data (array)
        bins (int): number of histogram bins, default=100
    
    Return:
        array: computed 1d probability distribution
    """
    data = normalization(data.flatten())
    prob = discrete_distribution(data, bins=bins)

    return prob

def partition_data(array, partition_size=500, seed=None):
    """Partition the into multiple groups with given size

    It will partition the data with same schema if seed is consistent for 
    each partitioning

    Args:
        array (array)
        partition_size(int): size of each partition group, default=500
        seed (int): random seed to partionting the data, default=0
    
    Returns:
        list: each partition group is a element in the list
    """
    data = array.copy()
    if seed != None:
        np.random.seed(seed)
    # Shuffle
    np.random.shuffle(data)
    
    # Set the upper bound
    upper_bound = len(data)+1 if len(data) % partition_size != 0 else len(data)

    groups = []
    for i in range(0, upper_bound, partition_size):
        groups.append(data[i: i+partition_size])
    
    return groups

def adaptivity(data1, data2, threshold, bins=100, partition_size=500, seed=0):
    """Compute the adaptivity metric value for two datasets

    First partition the data1 and data2 into multiple subgroups with equal size.
    Normalize the subgroups data, and then calculate the discrete probability
    distribution for them. Compute the JS Divergence among the subgroups of data1
    and data2. Denote the number of pairs that satisfy  the condition: JS Divergence
    < threshold as num_matches. The adaptivity value is computed by num_matches / 
    (number of the data2's subgroups).

    Args:
        threshold (float): the threshold of JS-Divergence
        bins (int): number of histogram bins, default=100
        partition_size (int): size of each partition group, default=500
        seed (int): random seed to partionting the data, default=0

    Returns:
        float: the adaptivity value
    """
    
    d1_groups = partition_data(data1, partition_size, seed)
    d2_groups = partition_data(data2, partition_size, seed)
    num_matches = 0
    for d1 in d1_groups:
        prob1 = data_to_probability(d1, bins)
        for d2 in d2_groups:
            prob2 = data_to_probability(d2, bins)
            
            jsd = jensen_shannon_divergence(prob1, prob2)
            if jsd < threshold:
                num_matches += 1
    adaptivity = num_matches / len(d2_groups)

    return adaptivity    
    
def f_score(precision, recall, beta=1):
    """Compute F beta score
    
    Args:
        precision (float): precision
        recall (float): recall
        beta (float): the weight of recall, default=1
    
    Returns:
        float: f score
    """
    
    if recall + precision*beta**2 == 0:
        return 0
    else:        
        return (1 + beta**2)*precision*recall / (recall + precision*beta**2)

def accuracy_metrics(act, exp):
    """Return precision and recall for a single query
        
    Args:
        act (set): actual result
        exp (set): expected result
    
    Returns:
        float, float: precision and recall
    """
    
    if act == exp:
        # Consider the edge case: act=[] exp=[]
        return [1, 1]

    precision = len(act.intersection(exp)) / len(act) if len(act) != 0 else 0
    recall = len(act.intersection(exp)) / len(exp) if len(exp) != 0 else 0

    return precision, recall

def evaluation(act_dict, exp_dict):
    """Evaluate the accuracy of a actual result dict

    The dictionary organized in the following format:
        {'query_table_id': ['retrieved_table_id']}
        Example: {'q1', :['q2','q2','q3'], 'q2': ['q1', 'q5']}
    
    Args:
        act_dict (defaultdict(set)): actual result
        exp_dict (defaultdict(set)): expected result
    
    Returns:
        array with size(1,4): [precision, recall, f1, f05]

    """

    size = len(act_dict)
    # Construct a nx4 array to store accuracy metric values, n is the number of 
    # results
    # col0: precision, col1: recall
    # col2: f1 score,  col3: f0.5 score
    performance_matrics = np.zeros((size, 4))
    
    index = 0

    if len(act_dict.keys()) == 0:
        # Edge case: act_dict={}
        return np.array([0, 0, 0, 0])

    for key in act_dict.keys():
        act = act_dict[key]
        exp = exp_dict[key]
        precision, recall = accuracy_metrics(act, exp)
        f1 = f_score(precision, recall)
        f05 = f_score(precision, recall, beta=0.5)
        
        performance_matrics[index, :] = (precision, recall, f1, f05)
        index += 1
    
    return np.average(performance_matrics, axis=0)


def word_tokenize(data):
    """Tokenize a series of sentences to a list of words

    Tokenize the sentence, then removing the punctuation, stopwords, 
    and then transfer to lowercase.

    Args: 
        data (df.Series): a series of sentences

    Return:
        list: a list of words
    """
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')

    cleaned_vals = data
    temp = []
    for x in cleaned_vals:
        # Remove punctuation and tokenize the data
        temp.extend(tokenizer.tokenize(str(x)))
    cleaned_vals = temp
    # Remove the stopwords
    cleaned_vals = [x.lower() for x in cleaned_vals if x.lower() not in stop_words]
    # Remove nan
    cleaned_vals = [x for x in cleaned_vals if x != 'nan']

    return cleaned_vals

def flatten_df(df):
    """Flatten the df to an array

        Args:
            df(pd.DataFrame): a dataframe
        
        Returns:
            an array
    """
    return df.values.flatten()


def jsd_for_word(d1, d2, ct3=None, size=300):
    """Compute the JS Divergence based on most common words

        Args:
            d1 (df.Series): a series of sentences
            d2 (df.Series): a series of sentences 
            ct3 (list): a user-specified dictionary. If it is not given, we will 
                generate a top-K dictionary from the d1 and d2. The list is 
                list(collections.Counter), k is equals to param. size
            size (int, optional): the size of dictionary, default=300
        
        Returns:
            float: JS Divergence between d1 and d2
    """
    ## TODO  code refactor needed
    # Use collections.Counter to count the occurrence of the words
    ct1 = Counter(i for i in d1)
    ct2 = Counter(i for i in d2)

    # Use the frequency of the words from d1 and d2 to construct a common space
    if ct3 == None:
        ct3 = ct1 + ct2
        # Sort the counter by frequency first, then by alphabetical
        ct3 = sorted(ct3.items(), key=lambda  item: (-item[1], item[0]))

    # the last position is used to handle the edge case:
    # if d1 and d2 don't share a common word, prob2[-1] will be set to 1, 
    # otherwise np.sum(prob2) = 0
    prob1 = np.zeros(size+1)
    prob2 = np.zeros(size+1)

    index = 0

    # assign the occurrence of the most common k words
    for record in ct3[: size]:
        word = record[0]
        prob1[index] = ct1[word]
        prob2[index] = ct2[word]
        index += 1

    # generate the probability distribution
    prob1 = prob1 / np.sum(prob1)
    if np.sum(prob2 != 0):
        prob2 = prob2 / np.sum(prob2)
    else:
        prob2[index] = 1
        
    return jensen_shannon_divergence(prob1, prob2)

def adaptivity_word(d1, d2, word1, word2, threshold, partition_size=300, seed=0):
    """Compute the adaptivity for word-based data
    Args:
        d1
        d2
        word1:
        word2:
        threshold:
        partition_size:
        seed:
    """
    ## TODO  code refactor needed
    d1_groups = partition_data(d1.values, partition_size, seed)
    d2_groups = partition_data(d2.values, partition_size, seed)
    num_matches = 0
    ct3 = list(generate_common_dict(word1, word2))
    for d_1 in d1_groups:
        d_1 = word_tokenize(d_1.flatten())
        for d_2 in d2_groups:
            d_2 = word_tokenize(d_2.flatten())
            jsd = jsd_for_word(d_1, d_2, ct3 )
            if jsd <= threshold:
                num_matches += 1
    adaptivity = num_matches / len(d2_groups)
    return adaptivity

def generate_common_dict(d1, d2):
    """Generate a dictionary by combining d1 and d2

    Args:
        d1 (list): a list of words
        d2 (list): a list of words
    
    Returns:
        Counter: combined dictionary
    """
    word1 = d1.copy()
    word1.extend(d2)
    
    c = Counter(i for i in word1)
    return c

def l2d_btw_domains(d1, d2, ct3=None, size=300):
    """Compute the L2D distance in word-based data
        Args:
            d1 (df.Series): a series of sentences
            d2 (df.Series): a series of sentences 
            size (int, optional): the size of dictionary, default=300
        Returns:
            float: JS Divergence between d1 and d2
    """
    ## TODO  code refactor needed
    # Use collections.Counter to count the occurrence of the words
    ct1 = Counter(i for i in d1)
    ct2 = Counter(i for i in d2)

    # Use the frequency of the words from d1 and d2 to construct a common space
    if ct3 == None:
        ct3 = ct1 + ct2
        # Sort the counter by frequency first, then by alphabetical
        ct3 = sorted(ct3.items(), key=lambda  item: (-item[1], item[0]))

    # the last position is used to handle the edge case:
    # if d1 and d2 don't share a common word, prob2[-1] will be set to 1, 
    # otherwise np.sum(prob2) = 0
    prob1 = np.zeros(size+1)
    prob2 = np.zeros(size+1)

    index = 0

    # assign the occurrence of the most common k words
    for record in ct3[: size]:
        word = record[0]
        prob1[index] = ct1[word]
        prob2[index] = ct2[word]
        index += 1

    # generate the probability distribution
    prob1 = prob1 / np.sum(prob1)
    if np.sum(prob2 != 0):
        prob2 = prob2 / np.sum(prob2)
    else:
        prob2[index] = 1
        
    return l2_distance(prob1, prob2)