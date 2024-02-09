# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from haystack.nodes import BM25Retriever
from haystack.nodes import BM25Retriever, SentenceTransformersRanker
from haystack import Pipeline
from sentence_transformers.cross_encoder import CrossEncoder

def retrieve_docs(pipe, query, threshold):
    docs = pipe.run(query)['documents']
    ret_docs = []
    for doc in docs:
        if doc.score > threshold:
            ret_docs.append(doc)
            # print('doc:\n', doc.content)
            # print('score:\n', doc.score)
    return ret_docs    

def get_unique_docs(docs):
    ret_docs = []
    doc_content=[]
    for d in docs:
        if d.content not in doc_content:
            ret_docs.append(d)
            doc_content.append(d.content)
    return ret_docs


def calculate_similarity_scores(model, query, docs):
    sentence_paris = []
    for d in docs:
        sentence_paris.append([query, d.content])
    scores = model.predict(sentence_paris)
    return scores


def sort_docs_by_scores(docs, scores):
    return [(x, y) for y, x in sorted(zip(scores, docs), reverse = True)]


def score_and_filter_docs(q, docs, model):
    scores = calculate_similarity_scores(model, q, docs)
    docs_scores = sort_docs_by_scores(docs, scores)
    final_docs = []
    final_scores = []
    for d, s in docs_scores:
        if s > 0.2:
            final_docs.append(d)
            final_scores.append(s)
            # print('{:.3f} {}'.format(s, d.content))
    return final_docs, final_scores



def retrieve_with_keywords(pipe, keywords, threshold):
    combined_kw = ""
    docs = []
    for kw in keywords:
        # k = kw[0]
        combined_kw += (kw +" ")
        # print('keyword: ', kw)
        docs.extend(retrieve_docs(pipe, kw, threshold))
        # print('-'*50)
    # print('combined keyword: ', combined_kw)
    docs.extend(retrieve_docs(pipe, combined_kw.strip(),threshold))
    docs = get_unique_docs(docs)
    print('{} docs retrieved with keywords'.format(len(docs)))
    return docs



class SparseBM25Retriever():
    """Retrieve the document database with BM25 sparse algorithm."""

    def __init__(self, document_store = None, top_k = 1, rerank_topk = 1, 
                 score_threshold = 0.8, injection = False):
        assert document_store is not None, "Please give a document database for retrieving."
        self.retriever = BM25Retriever(document_store=document_store, top_k=top_k)
        self.pipe = None
        if rerank_topk <= top_k: # activate ranker
            self.ranker = SentenceTransformersRanker(model_name_or_path="BAAI/bge-reranker-base", top_k=rerank_topk)
            self.pipe = Pipeline()
            self.pipe.add_node(component=self.retriever, name="BM25Retriever", inputs=["Query"])
            self.pipe.add_node(component=self.ranker, name="Ranker", inputs=["BM25Retriever"])
        self.score_threshold = score_threshold
        self.top_k = top_k
        self.rerank_topk = rerank_topk
        self.injection = injection

    def query_the_database(self, query):
        if self.pipe==None:
            documents = self.retriever.retrieve(query)
        # print('query: \n', query)
        else:
            documents = self.pipe.run(query=query)['documents']
            # print('documents: \n', documents)
        context = ""
        for doc in documents:
            # print('doc:\n', doc)
            score = doc.score
            if score > self.score_threshold:
                context = context + doc.content + "\n"
                print('doc: \n', doc.content)
                print('score:\n', doc.score)
        if len(context) >0:
            return context.strip()
        else:
            return ""
    

    def query_the_database_with_keywords(self, keywords, query, malicious_injection = ""):
        assert self.pipe!=None, "need to have a retriever+ranker pipeline to query with keywords"
        
        similarity_model = CrossEncoder("BAAI/bge-reranker-base")
        # retrieve docs with query + keywords
        docs = self.pipe.run(query)['documents']
        print('keywords: ', keywords)
        docs2 = retrieve_with_keywords(self.pipe, keywords, self.score_threshold)
        docs.extend(docs2)
        docs = get_unique_docs(docs)
        docs, scores = score_and_filter_docs(query, docs, similarity_model)
        # print('{} docs returned'.format(len(docs)))
        # print('-'*50)
        n = 0
        # border string
        context = "="*15+'\n'
        for doc, score in zip(docs, scores):
            # print('doc:\n', doc.content)
            # print('score:\n', score)
            # temp = "Document [{}]: {}\n".format(n+1, doc.content)
            temp = doc.content + '\n'
            context = context + temp
            n+=1
            if n==min(self.top_k, self.rerank_topk):
                break
        # border string
        context = context + '='*15+'\n'

        # indirect injection if any
        context = context + malicious_injection
        
        if len(context) >0:
            return context.strip()
        else:
            return ""


