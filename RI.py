import glob
import re
import os
import math
from nltk.data import normalize_resource_name
from nltk.tokenize import word_tokenize
from nltk.util import transitive_closure
import numpy as np

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

class GroundTruth:
    def __init__(self, id, doc_name, is_relevant):
        self.id = id
        self.doc_name = doc_name
        self.is_relevant = is_relevant

    def get_ground_truth(self):
        ground_truth = [self.id, self.doc_name, self.is_relevant]
        return ground_truth

class Query:
    def __init__(self, lang, num, title, desc, narr):
        self.lang = lang
        self.num = num 
        self.title = title
        self.desc = desc
        self.narr = narr

    def get_ground_truth(self):
        ground_truth = [self.id, self.doc_name, self.is_relevant]
        return ground_truth

def main():
    print('(1) Modelo Probabilistico\n(2) Modelo Vetorial\n')
    option = int(input("Digite a opção: "))

    if(option == 1 or option == 2):
        config_ground_truth()
        config_query_options()
        pre_processing(option)
    else:
        print('Opção inválida')
    
    return

def config_ground_truth():
    file_folder = 'FIRE2010/en.qrels.76-125.2010.txt'
    file = open(file_folder, "r")

    id = None
    doc_name = None
    is_relevant = None
    ground_truths = []


    lines = file.readlines()
    print(lines[0], "LAL")

    for line in lines:
        line_separated = line.split()

        id = line_separated[0]
        doc_name = line_separated[2]
        is_relevant = line_separated[3]

        ground_truth = GroundTruth(id, doc_name, is_relevant)
        ground_truths.append(ground_truth)

        #print("linha separada", line_separated)
         
        #print("ID: ", id, "doc_name: ", doc_name, "is_relevant: ", is_relevant)
    
    #print(ground_truths[0].id)

def config_query_options():
    file_folder = 'FIRE2010/en.topics.76-125.2010.txt'
    file = open(file_folder, "r")

    lang = None
    num = None
    title = None
    desc = None
    narr = None
    queries = []

    lines = file.readlines()
    #print(lines, "OPA")

    for line in lines:
        line_separated = line.split()

        if("<top" in line_separated):
            print(line_separated[1])
        
        if("<num>" in line_separated):
            print(line_separated[0])
        
        if("<title>" in line_separated):
            print(line_separated[0])
        
        if("<desc>" in line_separated):
            print("a")

        if("<narr>" in line_separated):
            print("a")
        
        # if(len(line_separated) < 0):
        #     #query = Query(lang, num, title, desc, narr)
        #     #queries.append(query)
            

        print("linha separada", line_separated)

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

def pre_processing(option):
    #file_folder = 'FIRE2010/en.doc.2010/TELEGRAPH_UTF8/2004_utf8/atleisure/*utf8'
    file_folder = 'data/*'
    dict_global = {}
    words_in_doc = {}
    docs_mapping = []
    N = 0

    for file in sorted(glob.glob(file_folder)):
        print(file)
        filename = file
        file = open(file, "r")
        text = file.read()
        text = remove_special_characters(text)
        words = word_tokenize(text)
        words = [word.lower() for word in words]
        finding_all_unique_words_and_freq(dict_global, words)
        idx = os.path.basename(filename)
        docs_mapping.append(idx)

        if(option == 1):
            words_in_doc[idx] = words
        elif(option == 2):
            words_in_doc[N] = words

        # print('Filtered tokens:', words)
        
        N += 1
        file.close()

    unique_words_all = sorted(set(dict_global.keys()))
    # print(unique_words_all)

    linked_list_data = {}

    for word in unique_words_all:
        linked_list_data[word] = LinkedList()

    for doc in words_in_doc.keys():
        words = words_in_doc[doc]
        for word in set(words):
            linked_list_data[word].add_doc(doc, words.count(word))

    linked_list_data['to'].print_list()

    if(option == 1):
        query = query_request()
        answer(words_in_doc, query, linked_list_data, N)

    elif(option == 2):
        #print(docs_mapping)
        m = np.zeros((len(unique_words_all), len(docs_mapping)))
        for i in range(len(unique_words_all)):
            word = unique_words_all[i]
            postings = linked_list_data[word].get_doclist()
            ni = linked_list_data[word].n_docs
            for node in postings:
                docID = node[0]
                freq = node[1]
                m[i, docID] = freq
           #print(word, m[i])

        #print(docs_mapping)
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
            #print(word, m[i])

        norm = np.sum(m**2, axis=0)
        norm = [math.sqrt(norm[i]) for i in range(len(norm))]
        #print(norm)


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def query_request():
    query = 'to do'
    query = remove_special_characters(query)
    query = word_tokenize(query.lower())

    return query

def answer(words_in_doc, query, linked_list_data, N):
    answer = {}
    for doc in words_in_doc.keys():
        words = words_in_doc[doc]
        common_words = intersection(query, words)
        score = 0
        for ki in common_words:
            if linked_list_data[ki].n_docs > N/2:
                score += math.log10((N+0.5)/(linked_list_data[ki].n_docs+0.5))
            else:
                score += math.log10((N-linked_list_data[ki].n_docs+0.5)/(linked_list_data[ki].n_docs+0.5))
        answer[doc] = score
    print('answer: ', answer)

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
    print('answer 2: ', answer)

    return linked_list_data

# Chamada da Main
if __name__ == "__main__":
    main()
