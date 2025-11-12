from typing import Union, Optional
import re

import numpy as np
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEndpoint
from langchain_ollama import ChatOllama
from langchain_core.messages import (
    AIMessage, SystemMessage,
    HumanMessage,
)
from langchain_core.language_models import BaseChatModel
import torch
from huggingface_hub import InferenceClient
from transformers import AutoProcessor, AutoModelForVision2Seq, pipeline

from src.simulator import EnvironmentState, QueryReturn


class LLMAgent:
    def __init__(self, model: str, system_task: str,
                 human_task: str, backend: str = "together",
                 device: Optional[str] = None,
                 model_kwargs: Optional[dict] = None):
        """Initialize the LLM agent.

        Args:
            model (str): The model name/path to use
            system_task (str): The system prompt describing the agent's role
            human_task (str): The task description for the agent
            backend (str): The backend to use ("together", "huggingface", "huggingface_remote", "openai", "ollama")
            device (Optional[str]): Device for local HuggingFace models (cuda/cpu)
            model_kwargs (Optional[dict]): Additional model configuration
        """
        self._system_task = system_task
        self._human_task = human_task
        self._first_message = True
        self._message_history = []

        if model_kwargs is None:
            model_kwargs = {}

        # Initialize LLM based on backend
        if backend == "together":
            self._llm = ChatTogether(
                model=model,
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
        elif backend == "huggingface":
            # Local HuggingFace model execution (text-only models)
            # Determine device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            # Create HuggingFace pipeline
            hf_pipeline = HuggingFacePipeline.from_model_id(
                model_id=model,
                task="text-generation",
                device=device,
                pipeline_kwargs={
                    "max_new_tokens": model_kwargs.get("max_new_tokens", 512),
                    "temperature": model_kwargs.get("temperature", 0.1),
                    "do_sample": model_kwargs.get("do_sample", True),
                },
                model_kwargs={
                    "torch_dtype": getattr(torch, model_kwargs.get("torch_dtype", "float16")),
                    "load_in_8bit": model_kwargs.get("load_in_8bit", False),
                    "load_in_4bit": model_kwargs.get("load_in_4bit", False),
                }
            )
            self._llm = ChatHuggingFace(llm=hf_pipeline)
        elif backend == "huggingface_vlm":
            # Local HuggingFace Vision-Language Model execution
            # Determine device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load VL model with image-text-to-text pipeline
            pipe = pipeline(
                "image-text-to-text",
                model=model,
                device=device if device == "cpu" else 0,  # 0 for cuda:0
                torch_dtype=getattr(torch, model_kwargs.get("torch_dtype", "float16")) if device == "cuda" else torch.float32,
                trust_remote_code=True,
                model_kwargs={
                    "load_in_8bit": model_kwargs.get("load_in_8bit", False),
                    "load_in_4bit": model_kwargs.get("load_in_4bit", False),
                }
            )

            # Wrap in HuggingFacePipeline
            hf_pipeline = HuggingFacePipeline(
                pipeline=pipe,
                pipeline_kwargs={
                    "max_new_tokens": model_kwargs.get("max_new_tokens", 512),
                    "temperature": model_kwargs.get("temperature", 0.1),
                    "do_sample": model_kwargs.get("do_sample", False),
                }
            )
            self._llm = ChatHuggingFace(llm=hf_pipeline)
        elif backend == "huggingface_remote":
            # Remote HuggingFace Inference API/Endpoints
            # Supports vision-language models like Qwen2.5-VL, Qwen3-VL, etc.
            hf_endpoint = HuggingFaceEndpoint(
                repo_id=model,
                task="text-generation",
                max_new_tokens=model_kwargs.get("max_new_tokens", 512),
                temperature=model_kwargs.get("temperature", 0.1),
                do_sample=model_kwargs.get("do_sample", True),
                # Add other inference parameters as needed
                timeout=model_kwargs.get("timeout", 120),
            )
            self._llm = ChatHuggingFace(llm=hf_endpoint)
        elif backend == "openai":
            self._llm = ChatOpenAI(
                model=model,
                temperature=0,
                max_retries=2,
            )
        elif backend == "ollama":
            # Self-hosted Ollama backend
            # Supports both text and vision-language models
            self._llm = ChatOllama(
                model=model,
                base_url=model_kwargs.get("base_url", "http://localhost:11434"),
                temperature=model_kwargs.get("temperature", 0),
                num_predict=model_kwargs.get("num_predict", model_kwargs.get("max_new_tokens", 512)),
                # Additional Ollama-specific parameters
                num_ctx=model_kwargs.get("num_ctx"),  # Context window size
                repeat_penalty=model_kwargs.get("repeat_penalty"),
                top_k=model_kwargs.get("top_k"),
                top_p=model_kwargs.get("top_p"),
                reasoning=False,
                verbose=True,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}. "
                           f"Choose from: together, huggingface, huggingface_vlm, huggingface_remote, openai, ollama")

    def _format_visible_objects(self, env_feedback: EnvironmentState) -> str:
        """Format visible objects list as a human-readable string.

        Args:
            env_feedback: The environment state containing visible objects

        Returns:
            str: Formatted string listing all visible objects with their types and IDs
        """
        if not env_feedback.visible_objects:
            return "Visible objects: None"

        visible_objects_list = [f"- type=\"{obj.object_type}\" id=\"{obj.object_id}\""
                                for obj in env_feedback.visible_objects]
        visible_objects_text = "Visible objects:\n" + "\n".join(visible_objects_list)
        return visible_objects_text

    def _print_thinking_tokens(self, ai_response_msg: AIMessage) -> None:
        """Print thinking/reasoning tokens from the AI response if available.

        Args:
            ai_response_msg: The AI response message
        """
        # Check for reasoning_content in additional_kwargs (Ollama reasoning mode)
        if hasattr(ai_response_msg, 'additional_kwargs'):
            additional = ai_response_msg.additional_kwargs
            if 'reasoning_content' in additional:
                print("\n" + "="*50)
                print("MODEL REASONING:")
                print("="*50)
                print(additional['reasoning_content'])
                print("="*50 + "\n")
                return

        # Check for thinking tokens in response_metadata
        if hasattr(ai_response_msg, 'response_metadata'):
            metadata = ai_response_msg.response_metadata

            # Different models store thinking differently
            # For some models, it's in 'thinking' field
            if 'thinking' in metadata:
                print("\n" + "="*50)
                print("MODEL THINKING:")
                print("="*50)
                print(metadata['thinking'])
                print("="*50 + "\n")

            # For some models, it's in 'reasoning' field
            elif 'reasoning' in metadata:
                print("\n" + "="*50)
                print("MODEL REASONING:")
                print("="*50)
                print(metadata['reasoning'])
                print("="*50 + "\n")

    def send_environment_feedback(
            self, env_feedback: Union[EnvironmentState, QueryReturn],
            execution_result: Optional[str] = None) -> AIMessage:
        """Send environment feedback to the LLM and get response.

        Args:
            env_feedback: The environment state after action execution
            execution_result: Optional result/error from code execution

        Returns:
            AIMessage: The LLM response containing Python code to execute
        """
        if self._first_message:
            self._first_message = False
            encoded_img = env_feedback.agent_camera_view

            # Create visible objects list with object_id
            visible_objects_text = self._format_visible_objects(env_feedback)
            print(visible_objects_text)

            self._message_history.extend([
                SystemMessage(content=self._system_task),
                HumanMessage(content=self._human_task),
                HumanMessage(content=visible_objects_text),
                HumanMessage(content=[
                    {"type": "text",
                     "text": "Here is a screenshot from the simulator"},
                    {"type": "image_url",
                     "image_url": {"url": encoded_img}}]),
            ])
            ai_response_msg = self._llm.invoke(self._message_history)
            self._message_history.append(ai_response_msg)

            # Print thinking tokens if available
            self._print_thinking_tokens(ai_response_msg)

            return ai_response_msg
        else:
            # self._message_history.pop(-3) # Remove the last 3 messages
            encoded_img = env_feedback.agent_camera_view
            env_feedback.agent_camera_view = ''

            # Build feedback message
            feedback_parts = [
                {"type": "text", "text": f"Environment State:\n{str(env_feedback)}"}
            ]

            # Add execution result if provided
            if execution_result:
                feedback_parts.append(
                    {"type": "text", "text": f"\n\nExecution Result:\n{execution_result}"}
                )

            # Create visible objects list (only for EnvironmentState, not QueryReturn)
            if isinstance(env_feedback, EnvironmentState):
                visible_objects_text = self._format_visible_objects(env_feedback)
                print(visible_objects_text)

                self._message_history.extend([
                    HumanMessage(content=feedback_parts),
                    HumanMessage(content=visible_objects_text),
                    HumanMessage(content=[
                        {"type": "text",
                         "text": "Here is a screenshot from the simulator"},
                        {"type": "image_url",
                         "image_url": {"url": encoded_img}}])
                ])
            else:
                # For QueryReturn, no visible_objects attribute
                self._message_history.extend([
                    HumanMessage(content=feedback_parts),
                    HumanMessage(content=[
                        {"type": "text",
                         "text": "Here is a screenshot from the simulator"},
                        {"type": "image_url",
                         "image_url": {"url": encoded_img}}])
                ])
            ai_response_msg = self._llm.invoke(self._message_history)
            self._message_history.append(ai_response_msg)

            # Print thinking tokens if available
            self._print_thinking_tokens(ai_response_msg)

            return ai_response_msg
