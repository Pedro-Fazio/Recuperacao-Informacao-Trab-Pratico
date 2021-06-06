import glob
import re
import os
import math
from nltk.tokenize import word_tokenize
import numpy as np

def remove_special_characters(text):
    regex = re.compile('[^a-zA-Z0-9\s]')
    text_returned = re.sub(regex, '', text)
    return text_returned

def finding_all_unique_words_and_freq(dict_global, words):
    for word in set(words):
        if word in dict_global.keys():
            dict_global[word] += words.count(word)
        else:
            dict_global[word] = words.count(word)

file_folder = 'data/*'
dict_global = {}
words_in_doc = {}
docs_mapping = []
N = 0
for file in sorted(glob.glob(file_folder)):
    filename = file
    file = open(file, "r")
    text = file.read()    
    text = remove_special_characters(text)
    words = word_tokenize(text)
    words = [word.lower() for word in words]
    finding_all_unique_words_and_freq(dict_global, words)
    idx = os.path.basename(filename)    
    docs_mapping.append(idx)
    words_in_doc[N] = words
    print('Filtered tokens:', words)
    N += 1
    file.close()

unique_words_all = sorted(set(dict_global.keys()))
print(unique_words_all)

class Node:
    def __init__(self, docId, freq):
        self.freq = freq
        self.doc = docId
        self.next = None
    
    def __str__(self):        
        return 'doc:' + str(self.doc) + ', freq:' + str(self.freq)

class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.n_docs = 0
    
    def print_list(self):
        aux = self.head
        while aux:
            print(aux)
            aux = aux.next
    
    def get_doclist(self):
        l = []
        aux = self.head
        while aux:
            l.append([aux.doc, aux.freq])
            aux = aux.next
        return l
    
    def add_doc(self, doc, freq):
        node = Node(doc, freq)        
        if self.head == None:
            self.head = node        
        else:
            self.tail.next = node
        self.tail = node
        self.n_docs += 1     

linked_list_data = {}
for word in unique_words_all:
    linked_list_data[word] = LinkedList()

for doc in words_in_doc.keys():
    words = words_in_doc[doc]
    for word in set(words):
        linked_list_data[word].add_doc(doc, words.count(word))        

linked_list_data['do'].print_list()

print(docs_mapping)
m = np.zeros((len(unique_words_all), len(docs_mapping)))
for i in range(len(unique_words_all)):
    word = unique_words_all[i]
    postings = linked_list_data[word].get_doclist()
    ni = linked_list_data[word].n_docs
    for node in postings:
        docID = node[0]
        freq = node[1]
        m[i, docID] = freq
    print(word, m[i])


print(docs_mapping)
m = np.zeros((len(unique_words_all), len(docs_mapping)))
for i in range(len(unique_words_all)):
    word = unique_words_all[i]
    postings = linked_list_data[word].get_doclist()
    ni = linked_list_data[word].n_docs
    idf = math.log2(N/ni)
    for node in postings:
        docID = node[0]
        freq = node[1]
        m[i, docID] = (1 + math.log2(freq))*idf
    print(word, m[i])

norm = np.sum(m**2, axis=0)
norm = [math.sqrt(norm[i]) for i in range(len(norm))]
print(norm)
