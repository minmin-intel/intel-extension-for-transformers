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

class SparseBM25Retriever():
    """Retrieve the document database with BM25 sparse algorithm."""

    def __init__(self, document_store = None, top_k = 1, rerank_topk = 1):
        assert document_store is not None, "Please give a document database for retrieving."
        self.retriever = BM25Retriever(document_store=document_store, top_k=top_k)
        self.pipe = None
        if rerank_topk <= top_k: # activate ranker
            self.ranker = SentenceTransformersRanker(model_name_or_path="BAAI/bge-reranker-base", top_k=rerank_topk)
            self.pipe = Pipeline()
            self.pipe.add_node(component=self.retriever, name="BM25Retriever", inputs=["Query"])
            self.pipe.add_node(component=self.ranker, name="Ranker", inputs=["BM25Retriever"])
        

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
            context = context + doc.content + "\n"
        return context.strip()
