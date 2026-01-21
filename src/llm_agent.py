"""LLM Agent with Stateless RAG-based Context Injection.

This module implements the "Stateless Execution with Context Injection" pattern
where each turn assembles a fresh prompt from:
1. Static information (system task, human task from YAML)
2. RAG memories (retrieved from vector store)
3. Current observation (last action result, visible objects, screenshot)

No conversation history is maintained - all context comes from RAG.
"""

from typing import Union, Optional, List
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
from src.vision import (
    VisionCoordinateEstimator, VLMBoundingBoxParser,
    BoundingBox, RelativeCoordinate
)


class LLMAgent:
    """LLM Agent with Stateless RAG-based Memory Architecture.

    This agent implements the "Stateless Execution with Context Injection" pattern:
    - No message history is maintained across turns
    - Each turn assembles a fresh prompt from static info + RAG memories + current obs
    - Memories are stored in a vector database and retrieved as needed
    - Fixed objects (non-pickupable) are persisted across episodes
    """

    def __init__(self, model: str, system_task: str,
                 human_task: str, backend: str = "together",
                 device: Optional[str] = None,
                 model_kwargs: Optional[dict] = None,
                 use_stateless_rag: bool = True,
                 memory_db_path: Optional[str] = "./agent_memory_db",
                 no_metadata_mode: bool = False):
        """Initialize the LLM agent.

        Args:
            model (str): The model name/path to use
            system_task (str): The system prompt describing the agent's role
            human_task (str): The task description for the agent
            backend (str): The backend to use ("together", "huggingface", "huggingface_remote", "openai", "ollama")
            device (Optional[str]): Device for local HuggingFace models (cuda/cpu)
            model_kwargs (Optional[dict]): Additional model configuration
            use_stateless_rag (bool): Whether to use stateless RAG pattern (default: True)
            memory_db_path (Optional[str]): Path for persistent memory storage (default: ./agent_memory_db)
            no_metadata_mode (bool): If True, use only VLM vision for object detection (no simulator metadata)
        """
        self._system_task = system_task
        self._human_task = human_task
        self._first_message = True
        self._current_step = 0
        self._use_stateless_rag = use_stateless_rag
        self._no_metadata_mode = no_metadata_mode

        # Initialize vision coordinate estimator for no-metadata mode
        if no_metadata_mode:
            self._vision_estimator = VisionCoordinateEstimator(
                image_width=800,
                image_height=600,
                fov_degrees=90.0,
                camera_height=1.57
            )
            self._bbox_parser = VLMBoundingBoxParser()
            self._detected_objects: List[RelativeCoordinate] = []

            # Dead Reckoning: Initialize internal position/rotation tracking
            # Start at origin (0, 0) facing North (0 degrees)
            self._dead_reckoning_position = {"x": 0.0, "y": 0.0, "z": 0.0}
            self._dead_reckoning_rotation = {"x": 0.0, "y": 0.0, "z": 0.0}

            # AI2-THOR default movement parameters
            self._move_distance = 0.25  # meters per move action
            self._rotation_degrees = 90.0  # degrees per rotation action

            print("\n[No-Metadata Mode] Using VLM vision only + Dead Reckoning")

        # Initialize memory components for stateless RAG pattern
        if use_stateless_rag:
            self._memory_store = MemoryStore(
                collection_name="agent_memory",
                persist_directory=memory_db_path  # Persist to disk
            )
            self._memory_writer = MemoryWriter(self._memory_store)
            self._memory_retriever = MemoryRetriever(self._memory_store)
            self._context_assembler = ContextAssembler(system_task, human_task)

            # Clean up transient memories, keep only fixed objects
            self._retain_only_fixed_objects()
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

    def _detect_objects_via_vlm(self, encoded_img: str) -> List[RelativeCoordinate]:
        """Use VLM to detect objects and estimate their positions (no metadata mode).

        This method sends the screenshot to the VLM with a special bounding box detection
        prompt, then parses the response and converts to relative coordinates.

        Args:
            encoded_img: Base64-encoded image from simulator

        Returns:
            List of RelativeCoordinate objects for detected objects
        """
        # VLM detection prompt - asks for bounding boxes in normalized coordinates
        detection_prompt = """Detect all objects in this image and provide bounding boxes.

For EACH visible object, output in this exact format:
ObjectType: [x_min, y_min, x_max, y_max]

Where coordinates are normalized (0.0 to 1.0):
- x_min: left edge (0=left, 1=right)
- y_min: top edge (0=top, 1=bottom)
- x_max: right edge
- y_max: bottom edge

Only list objects you can clearly see. Common kitchen objects:
Fridge, Microwave, Sink, StoveBurner, CounterTop, Cabinet, Drawer,
Apple, Potato, Tomato, Egg, Bread, Lettuce, Mug, Cup, Bowl, Plate,
Knife, Fork, Spoon, Pan, Pot, SaltShaker, PepperShaker, Toaster

Example output:
Fridge: [0.05, 0.10, 0.35, 0.95]
Apple: [0.60, 0.55, 0.70, 0.65]
Mug: [0.45, 0.60, 0.55, 0.75]

Now detect objects in the image:"""

        # Call VLM for detection
        detection_messages = [
            HumanMessage(content=[
                {"type": "text", "text": detection_prompt},
                {"type": "image_url", "image_url": {"url": encoded_img}}
            ])
        ]

        try:
            detection_response = self._llm.invoke(detection_messages)
            response_text = detection_response.content

            print("\n" + "="*50)
            print("[VLM Object Detection]")
            print("="*50)
            print(response_text)
            print("="*50 + "\n")

            # Parse bounding boxes from VLM response
            bboxes = self._bbox_parser.parse_normalized_bbox(response_text)

            if not bboxes:
                # Try pixel format as fallback
                bboxes = self._bbox_parser.parse_pixel_bbox(response_text)

            # Convert bounding boxes to relative coordinates
            coordinates = self._vision_estimator.batch_estimate(bboxes)

            # Store detected objects in memory
            for coord in coordinates:
                self._memory_writer.record_vision_based_object(
                    object_type=coord.object_type,
                    x_local=coord.x_local,
                    z_local=coord.z_local,
                    confidence=coord.confidence,
                    is_fixed=self._is_fixed_object(coord.object_type)
                )

            self._detected_objects = coordinates
            return coordinates

        except Exception as e:
            print(f"[VLM Detection Error] {e}")
            return []

    def _is_fixed_object(self, object_type: str) -> bool:
        """Check if an object type is fixed (non-pickupable).

        Args:
            object_type: The object type to check

        Returns:
            True if the object is typically fixed in place
        """
        fixed_objects = {
            "Fridge", "Microwave", "StoveBurner", "Stove", "Oven",
            "Sink", "SinkBasin", "CounterTop", "Cabinet", "Drawer",
            "DiningTable", "CoffeeTable", "SideTable", "Shelf",
            "Toaster", "CoffeeMachine", "GarbageCan", "Window", "Door",
            "LightSwitch", "Chair", "Sofa", "Bed", "Desk"
        }
        return object_type in fixed_objects

    def _format_detected_objects_text(self, coordinates: List[RelativeCoordinate]) -> str:
        """Format detected objects for prompt injection (no-metadata mode).

        Args:
            coordinates: List of detected object coordinates

        Returns:
            Formatted text describing detected objects and their positions
        """
        if not coordinates:
            return "Detected objects: None visible in current view."

        lines = ["Detected objects (vision-based estimation):"]
        for coord in coordinates:
            direction = "right" if coord.x_local >= 0 else "left"
            lines.append(
                f"- {coord.object_type}: ~{abs(coord.x_local):.1f}m {direction}, "
                f"~{coord.z_local:.1f}m ahead"
            )

        return "\n".join(lines)

    def _update_dead_reckoning(self, execution_result: Optional[str],
                                action_success: bool) -> None:
        """Update dead reckoning position based on executed action.

        Dead Reckoning: Track position internally by accumulating movements.
        Initial position is (0, 0) facing 0 degrees (North/+Z direction).

        Coordinate System:
        - X: Right (+) / Left (-)
        - Z: Forward (+) / Back (-)
        - Y rotation: 0=North(+Z), 90=East(+X), 180=South(-Z), 270=West(-X)

        Args:
            execution_result: The execution result string containing action info
            action_success: Whether the action succeeded
        """
        if not execution_result or not action_success:
            return

        import math

        # Current facing direction in radians
        facing_rad = math.radians(self._dead_reckoning_rotation["y"])

        # Parse action from execution result
        result_lower = execution_result.lower()

        # Movement actions
        if "move_ahead" in result_lower:
            # Move forward in facing direction
            self._dead_reckoning_position["x"] += self._move_distance * math.sin(facing_rad)
            self._dead_reckoning_position["z"] += self._move_distance * math.cos(facing_rad)

        elif "move_back" in result_lower:
            # Move backward (opposite of facing direction)
            self._dead_reckoning_position["x"] -= self._move_distance * math.sin(facing_rad)
            self._dead_reckoning_position["z"] -= self._move_distance * math.cos(facing_rad)

        elif "move_left" in result_lower:
            # Strafe left (perpendicular to facing)
            self._dead_reckoning_position["x"] -= self._move_distance * math.cos(facing_rad)
            self._dead_reckoning_position["z"] += self._move_distance * math.sin(facing_rad)

        elif "move_right" in result_lower:
            # Strafe right (perpendicular to facing)
            self._dead_reckoning_position["x"] += self._move_distance * math.cos(facing_rad)
            self._dead_reckoning_position["z"] -= self._move_distance * math.sin(facing_rad)

        # Rotation actions
        elif "rotate_left" in result_lower:
            # Rotate counter-clockwise (decrease Y rotation)
            self._dead_reckoning_rotation["y"] -= self._rotation_degrees
            # Normalize to 0-360
            self._dead_reckoning_rotation["y"] %= 360

        elif "rotate_right" in result_lower:
            # Rotate clockwise (increase Y rotation)
            self._dead_reckoning_rotation["y"] += self._rotation_degrees
            # Normalize to 0-360
            self._dead_reckoning_rotation["y"] %= 360

        # Round to avoid floating point drift
        self._dead_reckoning_position["x"] = round(self._dead_reckoning_position["x"], 2)
        self._dead_reckoning_position["z"] = round(self._dead_reckoning_position["z"], 2)

    def _get_dead_reckoning_state(self) -> tuple:
        """Get current dead reckoning position and rotation.

        Returns:
            Tuple of (position_dict, rotation_dict)
        """
        return (
            self._dead_reckoning_position.copy(),
            self._dead_reckoning_rotation.copy()
        )

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
        # [NO-METADATA MODE] - Update Dead Reckoning first
        # ============================================================
        if self._no_metadata_mode and not is_first_turn:
            # Update position/rotation based on the action that was just executed
            self._update_dead_reckoning(
                execution_result=execution_result,
                action_success=env_feedback.last_action_success if isinstance(env_feedback, EnvironmentState) else True
            )

        # ============================================================
        # [NO-METADATA MODE] - Detect objects via VLM
        # ============================================================
        detected_coordinates = []
        if self._no_metadata_mode and isinstance(env_feedback, EnvironmentState):
            encoded_img = env_feedback.agent_camera_view
            detected_coordinates = self._detect_objects_via_vlm(encoded_img)

        # ============================================================
        # [Step 1: SAVE PHASE] - Write observations to memory
        # ============================================================
        if isinstance(env_feedback, EnvironmentState):
            if self._no_metadata_mode:
                # Use dead reckoning position/rotation (NO metadata!)
                dr_position, dr_rotation = self._get_dead_reckoning_state()

                # Record observation with dead-reckoned position
                self._memory_writer.record_observation(
                    agent_position=dr_position,
                    agent_rotation=dr_rotation,
                    action_success=env_feedback.last_action_success,
                    error_message=env_feedback.error_message
                )

                # Debug: Print dead reckoning state
                print(f"\n[Dead Reckoning] Position: ({dr_position['x']:.2f}, {dr_position['z']:.2f}), "
                      f"Facing: {dr_rotation['y']:.0f}°")
            else:
                # Standard mode: use metadata for object detection
                action_taken = None
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
            if self._no_metadata_mode:
                # Pass dead reckoning position for retrieval context
                dr_position, dr_rotation = self._get_dead_reckoning_state()
                context = self._memory_retriever.retrieve_relevant_context_no_metadata(
                    task_description=self._human_task,
                    agent_position=dr_position,
                    current_step=self._current_step
                )
            else:
                context = self._memory_retriever.retrieve_relevant_context(
                    task_description=self._human_task,
                    env_state=env_feedback,
                    current_step=self._current_step
                )
            memory_text = self._memory_retriever.format_as_text(context)

        # ============================================================
        # [Step 3: ASSEMBLE PHASE] - Build fresh prompt
        # ============================================================
        if self._no_metadata_mode:
            # Pass dead reckoning state to assembler (NO metadata!)
            dr_position, dr_rotation = self._get_dead_reckoning_state()
            messages = self._context_assembler.assemble_messages_no_metadata(
                agent_position=dr_position,
                agent_rotation=dr_rotation,
                detected_objects=detected_coordinates,
                action_success=env_feedback.last_action_success if isinstance(env_feedback, EnvironmentState) else True,
                error_message=env_feedback.error_message if isinstance(env_feedback, EnvironmentState) else "",
                encoded_image=env_feedback.agent_camera_view,
                memory_text=memory_text,
                execution_result=execution_result,
                current_step=self._current_step,
                is_first_turn=is_first_turn
            )
            # Print detected objects for debugging
            detected_text = self._format_detected_objects_text(detected_coordinates)
            print(detected_text)
        else:
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

    def _retain_only_fixed_objects(self) -> None:
        """Clean up transient memories, keeping only fixed object locations.

        This method is called at startup to:
        - Delete action history (success/failure records)
        - Delete observation history (navigation path)
        - Delete relation records (may be stale)
        - Delete pickupable object locations (positions may have changed)

        Only non-pickupable objects (Fridge, Stove, Sink, etc.) are retained
        as their positions don't change between episodes.
        """
        print("\n[Memory System] Cleaning up transient memories, keeping fixed objects...")

        initial_count = self._memory_store.count()

        # 1. Delete action history (success/failure records from previous episodes)
        try:
            self._memory_store.delete_by_filter({"memory_type": "action"})
        except Exception:
            pass  # Ignore if no matching documents

        # 2. Delete observation history (agent's navigation path)
        try:
            self._memory_store.delete_by_filter({"memory_type": "observation"})
        except Exception:
            pass

        # 3. Delete relation records (objects may have been moved)
        try:
            self._memory_store.delete_by_filter({"memory_type": "relation"})
        except Exception:
            pass

        # 4. Delete pickupable object locations (Apple, Potato, etc. may have moved)
        try:
            self._memory_store.delete_by_filter({"is_pickupable": True})
        except Exception:
            pass

        final_count = self._memory_store.count()
        deleted_count = initial_count - final_count

        print(f"[Memory System] Cleanup complete. Deleted {deleted_count} transient memories.")
        print(f"[Memory System] Retained {final_count} fixed object locations.\n")

    def reset_memory(self) -> None:
        """Reset the memory store for a new episode.

        Only applicable when using stateless RAG mode.
        This clears all memories including fixed objects.
        Use _retain_only_fixed_objects() for selective cleanup.
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
