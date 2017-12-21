import email as em
import random
from bs4 import BeautifulSoup
import os
import pickle
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import string

punctuation_map = dict((ord(char), None) for char in string.punctuation)

def insert_elastic_search(all_data):
        es = Elasticsearch(timeout=100)
        index_name  = "spam_data"
        doc_type = "mails"
        create_index(es, index_name, doc_type)

        items = []
        count = 0
        for element in all_data:
        	count += 1
                item = {}
                item["data"] =  element["data"].strip()
                item["label"] = element["train_test"]
                item["split"] = element["spam_ham"] 

                items.append({'index': {'_index': index_name, '_type': doc_type, 
                        '_id': element["file_name"]}})    
                items.append(item)

                if(count %1000 == 0):  
                   es.bulk(index=index_name, body=items)
                   items = []
        es.bulk(index=index_name, body=items)

def create_index(es, index_name, doc_type):
        if es.indices.exists(index_name):
                        return 
        request_body = {
                        "settings": {
                                     "index": {
                                         "store": {
                                                  "type": "default"
                                                },
                                         "number_of_shards": 1,
                                         "number_of_replicas": 0
                                        }
                                }
                        } 
                        
        es.indices.create(index=index_name, body=request_body)
        es.indices.put_mapping(
                index=index_name,
                doc_type=doc_type,
                body={
                        doc_type: {
                                'properties': {
                                        'data': { 'type': 'string',
                                                          'store': True,
                                                          'index': 'analyzed',
                                                          'term_vector': 'with_positions_offsets_payloads'},
                                        'label': {'type': 'string'},
                                        'split': {'type': 'string'}
                                }
                        }
                })

def extract_data(data):
        value = ""
        if("Content-Type" in data and "image" in data["Content-Type"]):
                value = ""
        if data.is_multipart():
                value += extract_multipart_data(data)
        else:
                soup = BeautifulSoup(data.get_payload(), "html.parser")
                for script in soup(["script", "style"]):
                                        script.extract()          
                value += soup.get_text()
        return value.translate(punctuation_map)


def extract_multipart_data(data):
        value = ""
        for payload in data.get_payload():
                if payload.is_multipart():
                        value += extract_multipart_data(payload)
                elif "image" in payload["Content-Type"]:
                        value += ""
                else:
                        soup = BeautifulSoup(payload.get_payload(), "html.parser")
                        for script in soup(["script", "style"]):
                                                                 script.extract()          
                        value += soup.get_text()
        return value

spam_ham = {}
with open("trec07p/full/index") as doc:
        content = doc.readlines()
for element in content:
        element = element.strip()
        s_h, file_name =  element.split()
        spam_ham[file_name.replace("../data/", "")] = s_h

train_test = {}
t_t = random.sample(xrange(1, 75419), int(round(0.2*75419)))
no = 1
for file in spam_ham:
        if no in t_t:
                train_test[file] = "test"
        else:
                train_test[file] = "train"
        no += 1

all_data = []
no = 1
for file_name in os.listdir("trec07p/data/"):
        print no
        no += 1
        with open("trec07p/data/"+file_name, 'r') as item:
                data = unicode(item.read(), errors='ignore')
                data = em.message_from_string(data)
                subject = data["subject"] if data["subject"] else ""
                all_data.append({
                        "data" : subject + " " +extract_data(data),
                        "file_name" : file_name,
                        "train_test": train_test[file_name],
                        "spam_ham": spam_ham[file_name]})
with open("all_data_new.pkl", "wb")  as output:
        pickle.dump(all_data, output, pickle.HIGHEST_PROTOCOL)
insert_elastic_search(all_data) 




