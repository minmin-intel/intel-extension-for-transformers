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

import os
from .retrieval_base import Retriever
from .detector.intent_detection import IntentDetector
from .detector.statement_verifier import StatementVerifier
from .indexing.indexing import DocumentIndexing
from intel_extension_for_transformers.neural_chat.pipeline.plugins.prompt.prompt_template \
    import generate_qa_prompt, generate_prompt, generate_qa_prompt_v2
from nltk.tokenize import sent_tokenize
from multi_rake import Rake
from keybert import KeyBERT


def extract_keywords_rake(query):
    rake = Rake()
    keywords = rake.apply(query)
    keywords_list = []
    for kw in keywords:
        keywords_list.append(kw[0])
    return keywords_list[:3]

def extract_keywords_keybert(query):
    kw_model = KeyBERT(model='all-mpnet-base-v2')
    keywords = kw_model.extract_keywords(query, keyphrase_ngram_range=(1, 2), 
                                         stop_words='english', highlight=False,top_n=3)

    keywords_list= list(dict(keywords).keys())
    # print(keywords_list)
    return keywords_list

class Agent_QA():
    def __init__(self, persist_dir="./output", process=True, input_path=None,
                 embedding_model="hkunlp/instructor-large", max_length=2048, retrieval_type="dense",
                 document_store=None, top_k=1, rerank_topk=1, score_threshold=0.8,
                 search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5},
                 append=True, index_name="elastic_index_1", retrieve_with_keywords=False,
                 rag_sysm=None, template_context="rag_with_context_memory", template_no_context="rag_without_context",
                 asset_path="/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/assets"):
        self.model = None
        self.tokenizer = None
        self.retrieval_type = retrieval_type
        self.retriever = None
        self.intent_detector = IntentDetector()
        self.verifier = StatementVerifier()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.rag_sysm = rag_sysm # .txt file that contains system message for rag
        self.template_context = template_context
        self.template_no_context = template_no_context
        self.retrieve_with_keywords = retrieve_with_keywords
        
        if os.path.exists(input_path):
            self.input_path = input_path
        elif os.path.exists(os.path.split(os.path.split(os.path.split(script_dir)[0])[0])[0] \
                            + '/assets/docs/'):
            self.input_path = os.path.split(os.path.split(os.path.split(script_dir)[0])[0])[0] \
                            + '/assets/docs/'
        elif os.path.exists(os.path.join(asset_path, 'docs/')):
            self.input_path = os.path.join(asset_path, 'docs/')
        else:
            print("The given file path is unavailable, please check and try again!")
        if append:
            print('Adding documents to docstore')
            self.doc_parser = DocumentIndexing(retrieval_type=self.retrieval_type, document_store=document_store,
                                               persist_dir=persist_dir, process=process,
                                               embedding_model=embedding_model, max_length=max_length,
                                               index_name = index_name)
            self.db = self.doc_parser.KB_construct(self.input_path)
            print('Finished adding documents to docstore')
        else:
            # print("Make sure the current persist path is new!")
            if self.retrieval_type == 'dense':
                if os.path.exists(persist_dir):
                    if bool(os.listdir(persist_dir)):
                        print('Loading existing doc store...')
                        self.doc_parser = DocumentIndexing(retrieval_type=self.retrieval_type,
                                                    document_store=document_store,
                                                    persist_dir=persist_dir, process=process,
                                                    embedding_model=embedding_model,
                                                    max_length=max_length,
                                                    index_name=index_name)
                        self.db = self.doc_parser.load(self.input_path)
                        print('Finished loading docstore!')
                    else:
                        print('Did not find persistent database, creating new one...')
                        self.doc_parser = DocumentIndexing(retrieval_type=self.retrieval_type,
                                                        document_store=document_store,
                                                        persist_dir=persist_dir, process=process,
                                                        embedding_model=embedding_model, max_length=max_length,
                                                        index_name = index_name)
                        self.db = self.doc_parser.KB_construct(self.input_path)
            elif self.retrieval_type=='sparse':
                print('Loading existing elasticsearch doc store...')
                self.doc_parser = DocumentIndexing(retrieval_type=self.retrieval_type,
                                            document_store=document_store,
                                            persist_dir=persist_dir, process=process,
                                            embedding_model=embedding_model,
                                            max_length=max_length,
                                            index_name=index_name)
                self.db = self.doc_parser.load(self.input_path)
            else:
                raise ValueError('{} retrieval type is not supported!'.format(self.retrieval_type))
        self.retriever = Retriever(retrieval_type=self.retrieval_type, document_store=self.db, 
                                   top_k=top_k,rerank_topk=rerank_topk, score_threshold=score_threshold,
                                   search_type=search_type, search_kwargs=search_kwargs)



    def pre_llm_inference_actions(self, model_name, query, malicious_injection):
        intent = self.intent_detector.intent_detection(model_name, query)
        context = None
        if 'qa' not in intent.lower():
            print("Chat with AI Agent.")
            prompt = generate_prompt(query)
        else:
            print("Chat with QA agent.")
            if self.retriever:
                print('retrieving relevant context...')
                if self.retrieve_with_keywords == False:
                    context = self.retriever.get_context(query)

                else:
                    # extract keywords and retrieve with query + keywords
                    keywords = extract_keywords_keybert(query)
                    context = self.retriever.get_context_with_keywords(keywords, query, malicious_injection)
                

                if len(context)>0:
                    prompt = generate_qa_prompt(query, context, rag_sysm=self.rag_sysm, 
                                                    template_context=self.template_context,
                                                    template_no_context=self.template_no_context)
                else:
                    prompt = query
                # print('prompt for QA agent: ', prompt)
            else:
                print('did not find a retriever...')
                prompt = generate_prompt(query)
        return prompt, context
    
    # def post_llm_inference_actions(self, model_name, response, context):
    #     # split response by sentence
    #     # sentences = sent_tokenize(response)
    #     sentences = response.split('\n')
    #     # check if each sentence is supported by context
    #     verified_sentences = ""
    #     for s in sentences:
    #         if len(s)<5:
    #             continue
    #         if 'yes' in  self.verifier.verify(model_name, s+'\n', context).lower():
    #             verified_sentences += (s+"\n")
        
    #     # regenerate response 
    #     # simplest way: just return verified sentences
    #     return verified_sentences



