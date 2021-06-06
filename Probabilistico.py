import glob
import re
import os
import math
from nltk.tokenize import word_tokenize

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
    words_in_doc[idx] = words
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
            l.append(aux.doc)
            aux = aux.next
        return l
    
    def add_doc(self, doc, freq):
        node = Node(doc, words.count(word))        
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

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

query = 'to do'
query = remove_special_characters(query)
query = word_tokenize(query.lower())

answer = {}
for doc in words_in_doc.keys():
    words = words_in_doc[doc]
    common_words = intersection(query, words)
    score = 0
    for ki in common_words:
        score += math.log10((N+0.5)/(linked_list_data[ki].n_docs+0.5))
    answer[doc] = score
print(answer)

relevant_docs = ['doc1.txt', 'doc3.txt']
n_rel_docs = {}
for q in query:
    postings = linked_list_data[q].head
    n_rel_docs[q] = 0
    while postings:
        if postings.doc in relevant_docs: # TODO: optimize to O(n)
            n_rel_docs[q] += 1
        postings = postings.next
answer = {}
for doc in words_in_doc.keys():
    words = words_in_doc[doc]
    common_words = intersection(query, words)
    score = 0
    for ki in common_words:
        ni = linked_list_data[ki].n_docs
        R = len(relevant_docs)
        ri = n_rel_docs[ki]
        score += math.log10(((ri+0.5)*(N-ni-R+ri+0.5))/((R-ri+0.5)*(ni-ri+0.5)))
    answer[doc] = score
print(answer)