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

from intel_extension_for_transformers.neural_chat.prompts import PromptTemplate

"""The function for generating the target prompt."""

def generate_qa_prompt(query, context=None, history=None, rag_sysm=None):
    if context and history:
        conv = PromptTemplate("rag_with_context_memory")
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], context)
        conv.append_message(conv.roles[2], history)
        conv.append_message(conv.roles[3], None)
    elif context:
        conv = PromptTemplate("rag_with_context_memory")
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], context)
        conv.append_message(conv.roles[3], None)
    else:
        conv = PromptTemplate("rag_without_context")
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
    if rag_sysm:
        with open(rag_sysm, 'r') as f:
            sysm=f.read()
        # print('rag sysm: ', sysm)
        # print('setting rag system message...')
        conv.set_system_message(sysm+'\n')
    return conv.get_prompt()


def generate_qa_prompt_v2(query, context=None, history=None, rag_sysm=None):
    if context and history:
        conv = PromptTemplate("rag_with_context_memory_v2")
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], context)
        conv.append_message(conv.roles[2], history)
        conv.append_message(conv.roles[3], query)
        conv.append_message(conv.roles[4], None)
    elif context:
        conv = PromptTemplate("rag_with_context_memory_v2")
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], context)
        conv.append_message(conv.roles[3], query)
        conv.append_message(conv.roles[4], None)
    else:
        conv = PromptTemplate("rag_without_context")
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
    if rag_sysm:
        with open(rag_sysm, 'r') as f:
            sysm=f.read()
        # print('rag sysm: ', sysm)
        # print('setting rag system message...')
        conv.set_system_message(sysm+'\n')
    return conv.get_prompt()


def generate_prompt(query, history=None):
    if history:
        conv = PromptTemplate("rag_with_context_memory")
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[2], history)
        conv.append_message(conv.roles[3], None)
    else:
        conv = PromptTemplate("rag_without_context")
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def generate_intent_prompt(query):
    conv = PromptTemplate("intent")
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

def generate_verifier_prompt(query, context):
    conv = PromptTemplate("verifier")
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], context)
    conv.append_message(conv.roles[2], None)
    return conv.get_prompt()