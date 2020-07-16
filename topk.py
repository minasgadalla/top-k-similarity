# import libraries
import glob
import operator as op
import os
import pathlib
import re
import sys
from functools import reduce
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# function for combination calculation K<(N/2)
def ncr(n, r):
    r = min(r, n - r)  # necessary because combination cannot be negative
    numer = reduce(op.mul, range(n, n - r, -1), 1)  # calculate factorial with the reduce method
    denom = reduce(op.mul, range(1, r + 1), 1)  # calculate factorial with the reduce method
    return numer / denom


# calculates the cosine similarity and returns the top K similarity
def find_similar(tfidf_matrix, index, files, top_K):
    cosine_similarities = cosine_similarity(tfidf_matrix[index:index + 1],
                                            tfidf_matrix).flatten()  # calculate the cosine similarity
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [(files[index], cosine_similarities[index]) for index in related_docs_indices][0:top_K]


# remove words and specific special characters including numbers
def text_handler(text_i):
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
                 'his', 'himself'
        , 'her', 'hers', 'it', 'its', 'itself', 'they', 'then', 'and', 'the', 'of', 'in', 'at', 'on', 'μια', 'οι', 'πως',
                 'themselves', 'theirs', 'their', 'what', 'which', 'for','ce','cf','ca', 'that']  # remove these words from the text
    text = re.sub(r'\W', ' ', text_i)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'^b\s+', '', text)
    text = re.sub('[0-9]+', '', text)
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE) #remove the urls
    text = text.lower()
    text = text.split()
    resultwords = [word for word in text if word.lower() not in stopwords]
    result = ' '.join(resultwords)
    return result #return the optimized text


# print the whole Dataframe without ellipsis
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


# ask from user to insert the N number and check if it's valid
N_input = input("Enter number of documents N: ")
while True:
    try:  # check if the input is int
        N = int(N_input)
        print("Input number value is: ", N)
        print("N is valid...")
        break  # exit from while
    except ValueError:  # if the input is not an int ask again
        print("That's not an int! Please try again")
        N_input = input("Enter N: ") # re-enter N

# ask from user to insert the K number and check if it's valid
K_input = input("Enter the TOP-K value:  ")
while True:
    try:  # check if the input is int
        K = int(K_input)
        print("Input number value is: ", K)
        print("Checking value...")
        if (K < ncr(N, 2)):  # check if the input is less than NC2
            print("K is valid...")
            break  
        else:  # if the input is not less than NC2 then raise an exception
            print("K is NOT valid...try again")
            raise ValueError
    except ValueError:  # the input was invalid, repeat
        print("Input isn't an int or isn't less that the NC2. Please try again.")
        K_input = input("Enter the TOP-K value: ") # re-enter K

# print the imported values
print("Values Inported-- N: %d and K: %d" % (N, K))
print("------------------------------------------")

# ask for files, the user should input the folder
while True:
    try:
        input_directory = pathlib.Path(input('Please specify input folder, use "data" for samples: '))
        if input_directory.is_dir():  # check if the folder exists
            print("Folder exists...")
            print("------------------------------------------")
            break
        if not input_directory.is_dir():  # check if the folder doesn't exist
            print("Folder does not exist... please try again")
            print("------------------------------------------")
            print("Starting... Please wait!")
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise

# append evey file to a list
try:
    corpus = []  # create a list which will contain the files
    filenames = glob.glob(os.path.join(input_directory, '*.txt'))  # store the name of every file on a list
    for file in input_directory.glob('*.txt'):  # for every file in the directory
        with open(file, "r", encoding='utf-8', errors='ignore') as text_file:
            text = text_handler(text_file.read())  # optimise the text for usage
            corpus.append((file, text))  # add the text of the file to a list
    corpus_index = [n for n in corpus]
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise

# do tfidf calculation and print the resuls
try:
    tf = TfidfVectorizer()  # create an TfidfVectorizer item to calculate the tfidf matrix
    tfidf_matrix = tf.fit_transform([content for file, content in corpus])  # calculate the tfidf matrix for every file
    feature_names = tf.get_feature_names()
    itter = 0
    for i in corpus:
        if (itter > N - 1):  # if the iteration exceeds the number of documents imported from the user then quit
            break
        item = tfidf_matrix[itter]
        df = pd.DataFrame(item.T.todense(), index=feature_names,
                          columns=["tfidf_value"])  # generate a grid using Dataframd

        print("The similar files of %s are:" % filenames[itter])
        print(find_similar(tfidf_matrix, itter, filenames[:], K))

        print("The most unique words of this file are:")
        df = df.sort_values(by=["tfidf_value"], ascending=False)
        print_full(df.iloc[0:100, :])

        print("------------------------------------------")
        itter += 1
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise