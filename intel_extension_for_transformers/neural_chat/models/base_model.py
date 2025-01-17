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

from abc import ABC
from typing import List
import os, types
from fastchat.conversation import get_conv_template, Conversation
from ..config import GenerationConfig
from ..plugins import is_plugin_enabled, get_plugin_instance, get_registered_plugins, plugins
from ..utils.common import is_audio_file
from .model_utils import load_model, predict, predict_stream, MODELS
from ..prompts import PromptTemplate


def construct_parameters(query, model_name, device, config):
    params = {}
    params["prompt"] = query
    params["temperature"] = config.temperature
    params["top_k"] = config.top_k
    params["top_p"] = config.top_p
    params["repetition_penalty"] = config.repetition_penalty
    params["max_new_tokens"] = config.max_new_tokens
    params["do_sample"] = config.do_sample
    params["num_beams"] = config.num_beams
    params["model_name"] = model_name
    params["num_return_sequences"] = config.num_return_sequences
    params["bad_words_ids"] = config.bad_words_ids
    params["force_words_ids"] = config.force_words_ids
    params["use_hpu_graphs"] = config.use_hpu_graphs
    params["use_cache"] = config.use_cache
    params["ipex_int8"] = config.ipex_int8
    params["device"] = device
    return params

class BaseModel(ABC):
    """
    A base class for LLM.
    """

    def __init__(self):
        """
        Initializes the BaseModel class.
        """
        self.model_name = ""
        self.asr = None
        self.tts = None
        self.audio_input_path = None
        self.audio_output_path = None
        self.retriever = None
        self.retrieval_type = None
        self.safety_checker = None
        self.intent_detection = False
        self.cache = None
        self.device = None
        self.conv_template = None

    def match(self, model_path: str):
        """
        Check if the provided model_path matches the current model.

        Args:
            model_path (str): Path to a model.

        Returns:
            bool: True if the model_path matches, False otherwise.
        """
        return True

    def load_model(self, kwargs: dict):
        """
        Load the model using the provided arguments.

        Args:
            kwargs (dict): A dictionary containing the configuration parameters for model loading.

        Example 'kwargs' dictionary:
        {
            "model_name": "my_model",
            "tokenizer_name": "my_tokenizer",
            "device": "cuda",
            "use_hpu_graphs": True,
            "cpu_jit": False,
            "ipex_int8": False,
            "use_cache": True,
            "peft_path": "/path/to/peft",
            "use_deepspeed": False
            "hf_access_token": "user's huggingface access token"
        }
        """
        self.model_name = kwargs["model_name"]
        self.device = kwargs["device"]
        self.use_hpu_graphs = kwargs["use_hpu_graphs"]
        self.cpu_jit = kwargs["cpu_jit"]
        self.use_cache = kwargs["use_cache"]
        load_model(model_name=kwargs["model_name"],
                   tokenizer_name=kwargs["tokenizer_name"],
                   device=kwargs["device"],
                   use_hpu_graphs=kwargs["use_hpu_graphs"],
                   cpu_jit=kwargs["cpu_jit"],
                   ipex_int8=kwargs["ipex_int8"],
                   use_cache=kwargs["use_cache"],
                   peft_path=kwargs["peft_path"],
                   use_deepspeed=kwargs["use_deepspeed"],
                   optimization_config=kwargs["optimization_config"],
                   hf_access_token=kwargs["hf_access_token"])

    def predict_stream(self, query, config=None):
        """
        Predict using a streaming approach.

        Args:
            query: The input query for prediction.
            config: Configuration for prediction.
        """
        if not config:
            config = GenerationConfig()

        config.device = self.device
        config.use_hpu_graphs = self.use_hpu_graphs
        config.cpu_jit = self.cpu_jit
        config.use_cache = self.use_cache

        if is_audio_file(query):
            if not os.path.exists(query):
                raise ValueError(f"The audio file path {query} is invalid.")

        query_include_prompt = False
        self.get_conv_template(self.model_name, config.task)
        if self.conv_template.roles[0] in query and self.conv_template.roles[1] in query:
            query_include_prompt = True

        # plugin pre actions
        for plugin_name in get_registered_plugins():
            if is_plugin_enabled(plugin_name):
                plugin_instance = get_plugin_instance(plugin_name)
                if plugin_instance:
                    if hasattr(plugin_instance, 'pre_llm_inference_actions'):
                        if plugin_name == "asr" and not is_audio_file(query):
                            continue
                        if plugin_name == "retrieval":
                            response = plugin_instance.pre_llm_inference_actions(self.model_name, query)
                        else:
                            response = plugin_instance.pre_llm_inference_actions(query)
                        if plugin_name == "safety_checker" and response:
                            return "Your query contains sensitive words, please try another query."
                        else:
                            if response != None and response != False:
                                query = response
        assert query is not None, "Query cannot be None."

        if not query_include_prompt:
            query = self.prepare_prompt(query, self.model_name, config.task)
        response = predict_stream(**construct_parameters(query, self.model_name, self.device, config))

        def is_generator(obj):
            return isinstance(obj, types.GeneratorType)

        # plugin post actions
        for plugin_name in get_registered_plugins():
            if is_plugin_enabled(plugin_name):
                plugin_instance = get_plugin_instance(plugin_name)
                if plugin_instance:
                    if hasattr(plugin_instance, 'post_llm_inference_actions'):
                        if plugin_name == "safety_checker" and is_generator(response):
                            continue
                        response = plugin_instance.post_llm_inference_actions(response)

        # clear plugins config
        for key in plugins:
            plugins[key] = {
                "enable": False,
                "class": None,
                "args": {},
                "instance": None
            }

        return response

    def predict(self, query, config=None, malicious_injection=""):
        """
        Predict using a non-streaming approach.

        Args:
            query: The input query for prediction.
            config: Configuration for prediction.
        """
        if not config:
            config = GenerationConfig()

        config.device = self.device
        config.use_hpu_graphs = self.use_hpu_graphs
        config.cpu_jit = self.cpu_jit
        config.use_cache = self.use_cache

        if is_audio_file(query):
            if not os.path.exists(query):
                raise ValueError(f"The audio file path {query} is invalid.")

        query_include_prompt = False
        self.get_conv_template(self.model_name, config.task)
        # MH add
        # print('prompt template for predict: ',self.conv_template.get_prompt())
        #
        if self.conv_template.roles[0] in query and self.conv_template.roles[1] in query:
            print('query_include_prompt = True')
            query_include_prompt = True

        # plugin pre actions
        context = None
        for plugin_name in get_registered_plugins():
            if is_plugin_enabled(plugin_name):
                plugin_instance = get_plugin_instance(plugin_name)
                if plugin_instance:
                    if hasattr(plugin_instance, 'pre_llm_inference_actions'):
                        if plugin_name == "asr" and not is_audio_file(query):
                            continue
                        if plugin_name == "retrieval":
                            response, context = plugin_instance.pre_llm_inference_actions(self.model_name, query, malicious_injection)
                        else:
                            response = plugin_instance.pre_llm_inference_actions(query)
                        if plugin_name == "safety_checker" and response:
                            return "Your query contains sensitive words, please try another query."
                        else:
                            if response != None and response != False:
                                query = response
        assert query is not None, "Query cannot be None."

        if not query_include_prompt:
            print('query_include_prompt = False')
            query = self.prepare_prompt(query, self.model_name, config.task, config.system_message, config.single_turn)
        # LLM inference
        print('query passed into LLM: ', query)
        response = predict(**construct_parameters(query, self.model_name, self.device, config))

        # plugin post actions
        for plugin_name in get_registered_plugins():
            if is_plugin_enabled(plugin_name):
                plugin_instance = get_plugin_instance(plugin_name)
                if plugin_instance:
                    if hasattr(plugin_instance, 'post_llm_inference_actions'):
                        if plugin_name == "retrieval":
                            response = plugin_instance.post_llm_inference_actions(self.model_name, response, context)
                        else:
                            response = plugin_instance.post_llm_inference_actions(response)

        # clear plugins config
        # MH: why do this? 
        # If I want to continue to use RAG, I should have the retrieval plugin enabled.
        # for key in plugins:
        #     plugins[key] = {
        #         "enable": False,
        #         "class": None,
        #         "args": {},
        #         "instance": None
        #     }

        return response, context

    def chat_stream(self, query, config=None):
        """
        Chat using a streaming approach.

        Args:
            query: The input query for prediction.
            config: Configuration for prediction.
        """
        return self.predict_stream(query=query, config=config)

    def chat(self, query, config=None):
        """
        Chat using a non-streaming approach.

        Args:
            query: The input query for conversation.
            config: Configuration for conversation.
        """
        return self.predict(query=query, config=config)

    def get_default_conv_template(self, model_path: str) -> Conversation:
        """
        Get the default conversation template for the given model path.

        Args:
            model_path (str): Path to the model.

        Returns:
            Conversation: A default conversation template.
        """
        return get_conv_template("zero_shot")

    def get_conv_template(self, model_path: str, task: str = "") -> Conversation:
        """
        Get the conversation template for the given model path or given task.

        Args:
            model_path (str): Path to the model.
            task (str): Task type, one of [completion, chat, summarization].

        Returns:
            Conversation: A conversation template.
        """
        if self.conv_template:
            return
        if not task:
            self.conv_template = PromptTemplate(self.get_default_conv_template(model_path).name)
        else:
            if task == "completion":
                name = "alpaca_without_input"
            elif task == "chat":
                name = "neural-chat-7b-v2"
            elif task == "summarization":
                name = "summarization"
            else:
                raise NotImplementedError(f"Unsupported task {task}.")
            self.conv_template = PromptTemplate(name)

    def prepare_prompt(self, prompt: str, model_path: str, task: str = "", system_message: str="", single_turn: bool=False):
        self.get_conv_template(model_path, task)
        # print('system message: ', system_message)
        self.conv_template.set_system_message(system_message)
        # print('after setting sysm in prepare prompt:\n', self.conv_template.get_prompt())
        if single_turn == True:
            # remove [self.conv_template.roles[0], message] from self.conv_template.conv.messages
            # print('existing messages: ', self.conv_template.conv.messages)
            self.conv_template.conv.messages = [] # from test, we saw that the first time messages = []
            # print('after clearing messages: ', self.conv_template.get_prompt())

        self.conv_template.append_message(self.conv_template.roles[0], prompt)
        self.conv_template.append_message(self.conv_template.roles[1], None)
        # print('after prepare prompt: ', self.conv_template.get_prompt())
        return self.conv_template.get_prompt()

    def register_plugin_instance(self, plugin_name, instance):
        """
        Register a plugin instance.

        Args:
            instance: An instance of a plugin.
        """
        if plugin_name == "tts":
            self.tts = instance
        if plugin_name == "tts_chinese":
            self.tts_chinese = instance
        if plugin_name == "asr":
            self.asr = instance
        if plugin_name == "asr_chinese":
            self.asr_chinese = instance
        if plugin_name == "retrieval":
            self.retrieval = instance
        if plugin_name == "cache":
            self.cache = instance
        if plugin_name == "safety_checker":
            self.safety_checker = instance


# A global registry for all model adapters
model_adapters: List[BaseModel] = []


def register_model_adapter(cls):
    """Register a model adapter."""
    model_adapters.append(cls())


def get_model_adapter(model_name_path: str) -> BaseModel:
    """Get a model adapter for a model_name_path."""
    model_path_basename = os.path.basename(os.path.normpath(model_name_path)).lower()

    for adapter in model_adapters:
        if adapter.match(model_path_basename) and type(adapter) != BaseModel:
            return adapter

    raise ValueError(f"No valid model adapter for {model_name_path}")
