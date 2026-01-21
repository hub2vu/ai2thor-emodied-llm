"""LLM Agent with Stateless RAG-based Context Injection.

This module implements the "Stateless Execution with Context Injection" pattern
where each turn assembles a fresh prompt from:
1. Static information (system task, human task from YAML)
2. RAG memories (retrieved from vector store)
3. Current observation (last action result, visible objects, screenshot)

No conversation history is maintained - all context comes from RAG.
"""

from typing import Union, Optional
import re

import numpy as np
from langchain_openai import ChatOpenAI
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
from src.memory import (
    MemoryStore, MemoryWriter, MemoryRetriever, ContextAssembler
)


class LLMAgent:
    """LLM Agent with Stateless RAG-based Memory Architecture.

    This agent implements the "Stateless Execution with Context Injection" pattern:
    - No message history is maintained across turns
    - Each turn assembles a fresh prompt from static info + RAG memories + current obs
    - Memories are stored in a vector database and retrieved as needed
    """

    def __init__(self, model: str, system_task: str,
                 human_task: str, backend: str = "together",
                 device: Optional[str] = None,
                 model_kwargs: Optional[dict] = None,
                 use_stateless_rag: bool = True):
        """Initialize the LLM agent.

        Args:
            model (str): The model name/path to use
            system_task (str): The system prompt describing the agent's role
            human_task (str): The task description for the agent
            backend (str): The backend to use ("together", "huggingface", "huggingface_remote", "openai", "ollama")
            device (Optional[str]): Device for local HuggingFace models (cuda/cpu)
            model_kwargs (Optional[dict]): Additional model configuration
            use_stateless_rag (bool): Whether to use stateless RAG pattern (default: True)
        """
        self._system_task = system_task
        self._human_task = human_task
        self._first_message = True
        self._current_step = 0
        self._use_stateless_rag = use_stateless_rag

        # Initialize memory components for stateless RAG pattern
        if use_stateless_rag:
            self._memory_store = MemoryStore(collection_name="agent_memory")
            self._memory_writer = MemoryWriter(self._memory_store)
            self._memory_retriever = MemoryRetriever(self._memory_store)
            self._context_assembler = ContextAssembler(system_task, human_task)
        else:
            # Fallback to legacy mode with message history
            self._message_history = []

        if model_kwargs is None:
            model_kwargs = {}

        # Initialize LLM based on backend
        if backend == "huggingface":
            # Local HuggingFace model execution (text-only models)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

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
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            pipe = pipeline(
                "image-text-to-text",
                model=model,
                device=device if device == "cpu" else 0,
                torch_dtype=getattr(torch, model_kwargs.get("torch_dtype", "float16")) if device == "cuda" else torch.float32,
                trust_remote_code=True,
                model_kwargs={
                    "load_in_8bit": model_kwargs.get("load_in_8bit", False),
                    "load_in_4bit": model_kwargs.get("load_in_4bit", False),
                }
            )

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
            hf_endpoint = HuggingFaceEndpoint(
                repo_id=model,
                task="text-generation",
                max_new_tokens=model_kwargs.get("max_new_tokens", 512),
                temperature=model_kwargs.get("temperature", 0.1),
                do_sample=model_kwargs.get("do_sample", True),
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
            self._llm = ChatOllama(
                model=model,
                base_url=model_kwargs.get("base_url", "http://localhost:11434"),
                temperature=model_kwargs.get("temperature", 0),
                num_predict=model_kwargs.get("num_predict", model_kwargs.get("max_new_tokens", 512)),
                num_ctx=model_kwargs.get("num_ctx"),
                repeat_penalty=model_kwargs.get("repeat_penalty"),
                top_k=model_kwargs.get("top_k"),
                top_p=model_kwargs.get("top_p"),
                reasoning=False,
                verbose=True,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}. "
                           f"Choose from: huggingface, huggingface_vlm, huggingface_remote, openai, ollama")

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

            if 'thinking' in metadata:
                print("\n" + "="*50)
                print("MODEL THINKING:")
                print("="*50)
                print(metadata['thinking'])
                print("="*50 + "\n")

            elif 'reasoning' in metadata:
                print("\n" + "="*50)
                print("MODEL REASONING:")
                print("="*50)
                print(metadata['reasoning'])
                print("="*50 + "\n")

    def _extract_action_from_response(self, response_content: str) -> Optional[str]:
        """Extract action name from LLM response for memory tracking.

        Args:
            response_content: The LLM response text

        Returns:
            Extracted action name or None
        """
        # Look for simulator method calls
        patterns = [
            r'simulator\.(move_ahead|move_back|move_left|move_right)',
            r'simulator\.(rotate_left|rotate_right)',
            r'simulator\.(pick_object|put_object|open_object|close_object)',
            r'simulator\.(toggle_object_on|toggle_object_off)',
            r'simulator\.done'
        ]

        for pattern in patterns:
            match = re.search(pattern, response_content)
            if match:
                return match.group(0)

        return None

    def send_environment_feedback(
            self, env_feedback: Union[EnvironmentState, QueryReturn],
            execution_result: Optional[str] = None) -> AIMessage:
        """Send environment feedback to the LLM and get response.

        This method implements the Stateless Execution pattern:
        1. [Save Phase] Write new observations to memory
        2. [Retrieve Phase] Query relevant memories
        3. [Assemble Phase] Build fresh prompt from static + RAG + current obs
        4. [Invoke Phase] Get LLM response

        Args:
            env_feedback: The environment state after action execution
            execution_result: Optional result/error from code execution

        Returns:
            AIMessage: The LLM response containing Python code to execute
        """
        if self._use_stateless_rag:
            return self._send_feedback_stateless_rag(env_feedback, execution_result)
        else:
            return self._send_feedback_legacy(env_feedback, execution_result)

    def _send_feedback_stateless_rag(
            self, env_feedback: Union[EnvironmentState, QueryReturn],
            execution_result: Optional[str] = None) -> AIMessage:
        """Stateless RAG implementation of send_environment_feedback.

        Args:
            env_feedback: The environment state
            execution_result: Code execution result

        Returns:
            AIMessage: LLM response
        """
        is_first_turn = self._first_message
        self._first_message = False
        self._current_step += 1

        # ============================================================
        # [Step 1: SAVE PHASE] - Write observations to memory
        # ============================================================
        if isinstance(env_feedback, EnvironmentState):
            # Extract action from previous response (if any)
            action_taken = None  # Will be set after first turn

            # Write current observations to memory
            self._memory_writer.process_environment_state(
                env_state=env_feedback,
                action_taken=action_taken,
                execution_result=execution_result
            )

            # Track object relations if action succeeded
            if not is_first_turn and env_feedback.last_action_success:
                self._track_action_effects(env_feedback, execution_result)

        # ============================================================
        # [Step 2: RETRIEVE PHASE] - Query relevant memories
        # ============================================================
        memory_text = ""
        if isinstance(env_feedback, EnvironmentState):
            # Retrieve context relevant to task and current situation
            context = self._memory_retriever.retrieve_relevant_context(
                task_description=self._human_task,
                env_state=env_feedback,
                current_step=self._current_step
            )
            memory_text = self._memory_retriever.format_as_text(context)

        # ============================================================
        # [Step 3: ASSEMBLE PHASE] - Build fresh prompt
        # ============================================================
        messages = self._context_assembler.assemble_messages(
            env_state=env_feedback,
            memory_text=memory_text,
            execution_result=execution_result,
            current_step=self._current_step,
            is_first_turn=is_first_turn
        )

        # Print visible objects for debugging
        visible_objects_text = self._format_visible_objects(env_feedback)
        print(visible_objects_text)

        # Debug: Print memory context if available
        if memory_text:
            print("\n" + "="*50)
            print("RAG MEMORY CONTEXT:")
            print("="*50)
            print(memory_text)
            print("="*50 + "\n")

        # ============================================================
        # [Step 4: INVOKE PHASE] - Get LLM response
        # ============================================================
        ai_response_msg = self._llm.invoke(messages)

        # Print thinking tokens if available
        self._print_thinking_tokens(ai_response_msg)

        return ai_response_msg

    def _track_action_effects(self, env_feedback: EnvironmentState,
                               execution_result: Optional[str]) -> None:
        """Track effects of successful actions for memory.

        Args:
            env_feedback: Environment state after action
            execution_result: Execution result string
        """
        if not execution_result:
            return

        # Track pick actions
        if "pick_object" in execution_result.lower():
            # Find the picked object from execution result
            match = re.search(r'object_id="([^"]+)"', execution_result)
            if match:
                obj_id = match.group(1)
                obj_type = obj_id.split("|")[0] if "|" in obj_id else obj_id
                self._memory_writer.record_held_object(obj_id, obj_type)

        # Track put actions
        if "put_object" in execution_result.lower():
            match = re.search(r'target_id="([^"]+)"', execution_result)
            if match:
                target_id = match.group(1)
                target_type = target_id.split("|")[0] if "|" in target_id else target_id
                # Could track what was put where, but we'd need the object info

        # Track open/close actions
        if "open_object" in execution_result.lower():
            match = re.search(r'object_id="([^"]+)"', execution_result)
            if match:
                obj_id = match.group(1)
                obj_type = obj_id.split("|")[0] if "|" in obj_id else obj_id
                self._memory_writer.record_object_state_change(
                    obj_id, obj_type, "open", True
                )

        if "close_object" in execution_result.lower():
            match = re.search(r'object_id="([^"]+)"', execution_result)
            if match:
                obj_id = match.group(1)
                obj_type = obj_id.split("|")[0] if "|" in obj_id else obj_id
                self._memory_writer.record_object_state_change(
                    obj_id, obj_type, "open", False
                )

        # Track toggle actions
        if "toggle_object_on" in execution_result.lower():
            match = re.search(r'object_id="([^"]+)"', execution_result)
            if match:
                obj_id = match.group(1)
                obj_type = obj_id.split("|")[0] if "|" in obj_id else obj_id
                self._memory_writer.record_object_state_change(
                    obj_id, obj_type, "on", True
                )

        if "toggle_object_off" in execution_result.lower():
            match = re.search(r'object_id="([^"]+)"', execution_result)
            if match:
                obj_id = match.group(1)
                obj_type = obj_id.split("|")[0] if "|" in obj_id else obj_id
                self._memory_writer.record_object_state_change(
                    obj_id, obj_type, "on", False
                )

    def _send_feedback_legacy(
            self, env_feedback: Union[EnvironmentState, QueryReturn],
            execution_result: Optional[str] = None) -> AIMessage:
        """Legacy implementation using message history (for backward compatibility).

        Args:
            env_feedback: The environment state
            execution_result: Code execution result

        Returns:
            AIMessage: LLM response
        """
        if self._first_message:
            self._first_message = False
            encoded_img = env_feedback.agent_camera_view

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

            self._print_thinking_tokens(ai_response_msg)
            return ai_response_msg
        else:
            encoded_img = env_feedback.agent_camera_view
            env_feedback.agent_camera_view = ''

            feedback_parts = [
                {"type": "text", "text": f"Environment State:\n{str(env_feedback)}"}
            ]

            if execution_result:
                feedback_parts.append(
                    {"type": "text", "text": f"\n\nExecution Result:\n{execution_result}"}
                )

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

            self._print_thinking_tokens(ai_response_msg)
            return ai_response_msg

    def reset_memory(self) -> None:
        """Reset the memory store for a new episode.

        Only applicable when using stateless RAG mode.
        """
        if self._use_stateless_rag:
            self._memory_writer.reset()
            self._current_step = 0
            self._first_message = True
        else:
            self._message_history = []
            self._first_message = True

    def get_memory_stats(self) -> dict:
        """Get statistics about the memory store.

        Returns:
            Dictionary with memory statistics
        """
        if self._use_stateless_rag:
            return {
                "total_memories": self._memory_store.count(),
                "current_step": self._current_step,
                "visited_positions": len(self._memory_writer.get_visited_positions())
            }
        else:
            return {
                "message_history_length": len(self._message_history)
            }
