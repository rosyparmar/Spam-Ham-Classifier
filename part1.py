from elasticsearch import Elasticsearch
from elasticsearch import helpers
import pickle
from collections import defaultdict
from collections import OrderedDict
from sklearn import *
import numpy

es = Elasticsearch(timeout=100)
index_name  = "spam_data"
doctype = "mails"

def get_file_list(file_type):
    print "inside get file list"
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

def get_term_details(term):
    q = {
            "query": {
                 "function_score": {
                            "query": {
                                 "match": {
                                    "data": term
                            }
                        },
                        "functions": [
                            {
                                "script_score": {
                                    "lang": "groovy",
                                    "script_id": "getTF",
                                    "params": {
                                        "field": "content",
                                        "term": term
                                    }
                                }
                            }
                        ],
                        "boost_mode": "replace"
                    }
            },
            "fields": ["_id", "label"],
        }
    res1=es.search(index=index_name ,body=q, scroll='1m')
    scroll_data = [] 
    scroll_size=res1['hits']['total']
    while (scroll_size > 0):
        scroll_id = res1['_scroll_id']
        scroll_data += res1['hits']['hits']
        res1 = es.scroll(scroll_id=scroll_id, scroll='60s')
        scroll_size = len(res1['hits']['hits'])

    count = 0

    term_details = {}
    for doc in scroll_data:
        term_details[doc["_id"]]=doc["_score"]

    return term_details


def get_spam_file_list(spam_list):
    term_details = {}
    count = 0
    for term in spam_list:
        print count
        term_details[term] = get_term_details(term)
        count = count +1 
    return term_details

def get_labels():
    all_labels = {}
    with open("trec07p/full/index") as document:
        content = document.readlines()
    for line in content:
        spam_ham, file_name = line.strip().split()
        file_name = file_name.replace("../data/", "")
        all_labels[file_name] = 1 if spam_ham == "spam" else 0
    return all_labels

def get_spam_list():
    temp_spam_list = []
    with open("spam_words_own.txt") as content:
        temp_spam_list = content.readlines()
    spam_list = []
    for spam in temp_spam_list:
        spam_list.append(spam.strip())
    return spam_list

def generate_features(file_list, spam_list, spam_file_list):
    features = numpy.zeros(shape=(
        len(file_list), len(spam_list)))

    no_row = 0
    for file in file_list:
        new_row = []
        for word  in spam_list:
            if file in spam_file_list[word]:
                new_row.append(
                    spam_file_list[term][file])
            else:
                new_row.append(0)
        features[no_row] = new_row
        no_row += 1
    return features

def generate_labels(file_list, real_labels):
    labels = numpy.zeros(len(file_list))
    no_rows = 0
    for file in file_list:
        labels[no_rows] = real_labels[file]
        no_rows += 1
    return labels

def write_to_file(spam_ham, file_list):
    count = 0
    details = []
    for i in range(0, len(file_list), 1):
        details.append((file_list[0], spam_ham[count][1],
            spam_ham[count][0]))
    s_details = sorted(details, key=lambda tup: tup[1], reverse=True)

    with open("file_name", "wb") as value:
        for detail in s_details:
            value.write(detail[0]+" "+str(detail[1])+" "+str(detail[2]))
            value.write("\n")

def do_part1():
    train_file_list = get_file_list("train")
    print len(train_file_list)
    test_file_list  = get_file_list("test")
    print len(test_file_list)
    spam_list       = get_spam_list()
    spam_file_list = get_spam_file_list(spam_list)
    print "after get spam list"
    labels = get_labels()

    feature_train = generate_features(train_file_list, spam_list,
     spam_file_list)
    print "after feature train"
    #return
    labels_train  = generate_labels(train_file_list, labels)
    print "after labels train"

    feature_test = generate_features(test_file_list, spam_list,
     spam_file_list)
    labels_test  = generate_labels(test_file_list, labels)
    print "going to model"
    LR = linear_model.LogisticRegression()

#    LR = tree.DecisionTreeClassifier()
    LR.fit(feature_train, labels_train)
    print "after model"

    predict_labels_train = LR.predict(feature_train)
    
    count = 0
    for i in range(0, len(predict_labels_train), 1):
        if(predict_labels_train[i] == 
            labels_train[i]):
            count += 1
    print "accuracy in train part 1 ",
    print float(count)/len(labels_train)


    predict_labels_test  = LR.predict(feature_test)

    count = 0
    for i in range(0, len(predict_labels_test), 1):
        if(predict_labels_test[i] == 
            labels_test[i]):
            count += 1
    print "accuracy in test part 1 ",
    print float(count)/len(labels_test)

    test_spam_ham  = LR.predict_proba(feature_train)
    train_spam_ham = LR.predict_proba(feature_test)

    write_to_file(test_spam_ham, test_file_list)
    write_to_file(train_spam_ham, train_file_list)

do_part1()









