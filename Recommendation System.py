
# coding: utf-8

# In[1]:



# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    res=[]
    for i in range(0,len(movies.movieId)):
        abc=tokenize_string(movies.genres[int(i)])
        res.append(abc)
    movies["tokens"]=res
    return movies

def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    abc=set()
    for i in movies.tokens:
        for j in i:
            abc.add(j)

    vocab={}
    c=0
    for i in sorted(abc):
        vocab[i]=c
        c=c+1
    vocab

    #making list of dict for each movie
    lod=[]
    for i in range(0,len(movies.movieId)):
        temp={}
        for j in movies.tokens[i]:
            if j in temp:
                temp[j]=temp[j]+1
            else:
                temp[j]=1
        lod.append(temp)

    doc_count={}
    for i in sorted(abc):
        count=0
        for j in lod:
            if i in j:
                count=count+1
        doc_count[i]=count


    countdict={}
    csr_list=[]
    for i in lod:
        data=[]
        col=[]
        row=[]
        maxval=max([v for k,v in i.items()])
        for k,v in i.items():
            val=v/maxval *math.log10(len(movies.movieId)/doc_count[k])
            data.append(val)
            col.append(vocab[k])
            row.append(0)
        x = csr_matrix((data, (row, col)),shape=(1,len(vocab)))
        csr_list.append(x)
    movies["features"]=csr_list
    return tuple((movies,vocab))



def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    c=b.transpose()
    deno= (math.sqrt(sum((a.data)**2))) * (math.sqrt(sum((b.data)**2)))
    val = a.dot(c)/deno
    return val[0,0]
    
    
    
def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    result=[]
    for r_test in ratings_test.itertuples():
        weighted_sum=0
        weight_sum=0
        arr_notpos=[]
        flag=False
        for r_train in ratings_train.itertuples():
            if r_test.userId ==r_train.userId:
                arr_notpos.append(r_train.rating)
                val = cosine_sim(movies.features[movies.movieId==[r_train.movieId]].tolist()[0], movies.features[movies.movieId==[r_test.movieId]].tolist()[0])
                if val>0:
                    weighted_sum+=(val*r_train.rating)
                    #arr_pos.append(val*r_train["rating"])
                    weight_sum+=val
                    flag=True
        if flag==True:
            result.append((weighted_sum/weight_sum))
            #print((weighted_sum/weight_sum))
        else:
            result.append(np.array(arr_notpos).mean())
            #print(np.array(arr_notpos).mean())
    return np.array(result)

def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()





