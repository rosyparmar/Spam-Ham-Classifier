from collections import OrderedDict
from sklearn import linear_model
from collections import defaultdict
from scipy.sparse import coo_matrix
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import numpy as np
import pickle
import os
import time

es = Elasticsearch(timeout=100)
index_name  = "spam_data"
doctype = "mails"

def get_all_file_list():
    print "inside get file list"
    all_files = []
    query = {
                "query" : { 
                    "match_all" : {}
                        
                },
                "fields": ["_id"]
            }
    response = helpers.scan(client=es, query=query, index=index_name, doc_type=doctype)
    for hit in response:
        all_files.append(hit["_id"])
    return all_files

def get_labels():
    all_labels = {}
    with open("trec07p/full/index") as document:
        content = document.readlines()
    for line in content:
        spam_ham, file_name = line.strip().split()
        file_name = file_name.replace("../data/", "")
        all_labels[file_name] = 1 if spam_ham == "spam" else 0
    return all_labels

def get_file_list(file_type):
    all_files = []
    query = {
                "query" : { 
                    "match" : 
                        { "label" : file_type}
                },
                "fields": ["_id"]
            }
    response = helpers.scan(client=es, query=query, index=index_name, doc_type=doctype)
    for hit in response:
        all_files.append(hit["_id"])
    return all_files


def get_term_vectors(file_name):
    term_vectors = es.termvectors(index = index_name, doc_type = doctype, id=file_name, field_statistics = False,
                        offsets = False, positions = False, payloads = False )
    if ("term_vectors" in term_vectors and
            "content" in term_vectors["term_vectors"] and
            "terms" in term_vectors["term_vectors"]["content"]):
        return term_vectors["term_vectors"]["content"]["terms"]
    else:
        return {}

def generate_features(file_list, file_indexes, labels, 
    term_vectors, term_indexes):
    
    row    = []
    column = []
    data   = []

    for fil in file_list:
        term_vector = term_vectors[fil]
        for term in term_vector:
            row.append(file_indexes[fil])
            column.append(term_indexes[term])
            data.append(term_vector[term])
            #data.append(1)
    features = coo_matrix((np.array(data), 
                (np.array(row), np.array(column))), 
                shape=(len(file_list), len(term_indexes)))
    return features

def generate_labels(file_list, real_labels):
    labels = numpy.zeros(len(file_list))
    no_rows = 0
    for file in file_list:
        labels[no_rows] = real_labels[file]
        no_rows += 1
    return labels


def get_file_indexes(file_list):
    file_indexes = {}
    count = 0
    for file in file_list:
        file_indexes[file] = count
        count += 1
    return file_indexes

def get_term_indexes(term_vectors):
    term_indexes = {}
    if (os.path.exists("all_term_indexes.pkl")):
            with open('all_term_indexes.pkl', 'rb') as input:
                all_term_vectors = pickle.load(input)
                return all_term_vectors

    for file in term_vectors:
        for word in term_vectors[file]:
            if word not in term_indexes:
                term_indexes[word] = count
                count += 1

    with open('all_term_indexes.pkl', 'wb') as out:
            pickle.dump(term_indexes, out, pickle.HIGHEST_PROTOCOL)

    return term_indexes


def get_all_term_vectors(file_list):

    all_term_vectors = {}
    if (os.path.exists("all_term_vectors.pkl")):
            with open('all_term_vectors.pkl', 'rb') as input:
                all_term_vectors = pickle.load(input)
            return all_term_vectors

    for file in file_list:
        print file
        all_term_vectors[file] = get_term_vectors(file_list)
        time.sleep(0.01)

    with open('all_term_vectors.pkl', 'wb') as out:
            pickle.dump(all_term_vectors, out, pickle.HIGHEST_PROTOCOL)
    return all_term_vectors

def do_part2():
    
    file_list = get_all_file_list()
    train_file_list = get_file_list("train")
    test_file_list = get_file_list("test")
    all_labels = get_labels()

    term_vectors = get_all_term_vectors(file_list)
    term_indexes = get_term_indexes(term_vectors)
    train_file_indexes = get_file_indexes(train_file_list)
    test_file_indexes = get_file_indexes(test_file_list)

    train_features = generate_features(train_file_list, train_file_indexes, all_labels,
        term_vectors, term_indexes)
    train_labels   = generate_labels(train_file_list, all_labels)

    test_features =  generate_features(test_file_list, test_file_indexes, all_labels,
        term_vectors, term_indexes)
    test_labels   = generate_labels(test_file_list, all_labels)


    lr = linear_model.LogisticRegression()
    lr.fit(train_feature, train_labels)

    train_predict_labels  = lr.predict(train_feature)
    test_predict_labels  = lr.predict(test_features)

    print "part2 train accuracy ",
    print self.check_percentage(train_predict_labels, train_labels)

    print "part2 test accuracy ",
    print self.check_percentage(test_predict_labels, test_labels)

    test_spam_ham  = LR.predict_proba(feature_train)
    train_spam_ham = LR.predict_proba(feature_test)

    write_to_file(test_spam_ham, test_file_list)
    write_to_file(train_spam_ham, train_file_list)

do_part2()
        

