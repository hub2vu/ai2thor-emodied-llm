"""ContextAssembler - Assembles stateless prompts for each turn.

This module is the core of the Stateless Execution with Context Injection pattern.
It creates a fresh message list every turn by combining:
1. Static information (system prompt, task description)
2. Dynamic RAG memories (retrieved from vector store)
3. Current observation (last action result, visible objects, screenshot)
"""

from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.simulator import EnvironmentState, SimulatorObject

if TYPE_CHECKING:
    from src.vision import RelativeCoordinate


class ContextAssembler:
    """Assembles stateless prompts for LLM invocation.

    Each turn, this class creates a completely fresh message list containing:
    - System Prompt (static, from YAML config)
    - Task Definition (static, from YAML config)
    - Relevant Memories (dynamic, from RAG retrieval)
    - Current Observation (dynamic, real-time sensor data)

    This eliminates the need for maintaining conversation history and prevents
    context window explosion.
    """

    def __init__(self, system_task: str, human_task: str):
        """Initialize the context assembler.

        Args:
            system_task: The system prompt describing the agent's role and rules
            human_task: The task description (goal)
        """
        self._system_task = system_task
        self._human_task = human_task

    def assemble_messages(self,
                          env_state: EnvironmentState,
                          memory_text: Optional[str] = None,
                          execution_result: Optional[str] = None,
                          current_step: int = 0,
                          is_first_turn: bool = False) -> List[Union[SystemMessage, HumanMessage]]:
        """Assemble a fresh message list for LLM invocation.

        This method creates the messages from scratch every turn, implementing
        the stateless execution pattern.

        Args:
            env_state: Current environment state
            memory_text: Formatted text from RAG memory retrieval (optional)
            execution_result: Result/error from last code execution (optional)
            current_step: Current step number
            is_first_turn: Whether this is the first turn

        Returns:
            List of messages ready for LLM invocation
        """
        messages = []

        # ============================================================
        # 1. STATIC INFORMATION (from YAML - injected every turn)
        # ============================================================
        messages.append(SystemMessage(content=self._system_task))
        messages.append(HumanMessage(content=f"# Your Goal:\n{self._human_task}"))

        # ============================================================
        # 2. RAG MEMORIES (dynamic - retrieved from vector store)
        # ============================================================
        if memory_text and memory_text.strip():
            memory_section = self._format_memory_section(memory_text)
            messages.append(SystemMessage(content=memory_section))

        # ============================================================
        # 3. CURRENT OBSERVATION (dynamic - real-time sensor data)
        # ============================================================
        current_obs = self._format_current_observation(
            env_state=env_state,
            execution_result=execution_result,
            current_step=current_step,
            is_first_turn=is_first_turn
        )

        # Combine text observation with screenshot
        encoded_img = env_state.agent_camera_view
        messages.append(HumanMessage(content=[
            {"type": "text", "text": current_obs},
            {"type": "image_url", "image_url": {"url": encoded_img}}
        ]))

        return messages

    def _format_memory_section(self, memory_text: str) -> str:
        """Format the memory section for injection.

        Args:
            memory_text: Raw memory text from retriever

        Returns:
            Formatted memory section string
        """
        return f"""# Relevant Memories (from previous observations)
(This information was observed earlier. Objects may have moved since then.)

{memory_text}

---"""

    def _format_current_observation(self,
                                     env_state: EnvironmentState,
                                     execution_result: Optional[str],
                                     current_step: int,
                                     is_first_turn: bool) -> str:
        """Format the current observation section.

        Args:
            env_state: Current environment state
            execution_result: Last execution result (optional)
            current_step: Current step number
            is_first_turn: Whether this is the first turn

        Returns:
            Formatted observation string
        """
        sections = []

        # Header
        sections.append(f"# Current Observation (Step {current_step})")
        sections.append("(This is what you see RIGHT NOW.)\n")

        # Last Action Result (except on first turn)
        if not is_first_turn:
            if env_state.last_action_success:
                action_status = "SUCCESS"
            else:
                action_status = f"FAILED - {env_state.error_message}"

            sections.append(f"## Last Action Result: {action_status}")

            if execution_result:
                sections.append(f"Execution Output: {execution_result}")

            sections.append("")

        # Agent State
        pos = env_state.agent_position
        rot = env_state.agent_rotation
        sections.append(f"## Agent State:")
        sections.append(f"- Position: ({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f})")
        sections.append(f"- Rotation: {rot['y']:.0f} degrees")
        sections.append("")

        # Visible Objects
        sections.append("## Visible Objects:")
        if env_state.visible_objects:
            for obj in env_state.visible_objects:
                sections.append(f"- type=\"{obj.object_type}\" id=\"{obj.object_id}\"")
        else:
            sections.append("- (None visible)")
        sections.append("")

        # Instructions
        sections.append("## Instructions:")
        sections.append("Based on your Goal, Memories, and Current Observation above,")
        sections.append("generate Python code for your next action.")
        sections.append("")
        sections.append("Below is a screenshot from the simulator:")

        return "\n".join(sections)

    def assemble_messages_no_metadata(self,
                                       agent_position: Dict[str, float],
                                       agent_rotation: Dict[str, float],
                                       detected_objects: List["RelativeCoordinate"],
                                       action_success: bool,
                                       error_message: str,
                                       encoded_image: str,
                                       memory_text: Optional[str] = None,
                                       execution_result: Optional[str] = None,
                                       current_step: int = 0,
                                       is_first_turn: bool = False) -> List[Union[SystemMessage, HumanMessage]]:
        """Assemble messages for no-metadata mode using dead reckoning and vision.

        In no-metadata mode:
        - Position/rotation comes from Dead Reckoning (not simulator metadata)
        - Objects come from VLM detection (not visible_objects metadata)

        Args:
            agent_position: Dead reckoning estimated position
            agent_rotation: Dead reckoning estimated rotation
            detected_objects: List of RelativeCoordinate from vision detection
            action_success: Whether last action succeeded
            error_message: Error message if action failed
            encoded_image: Base64-encoded screenshot
            memory_text: Formatted text from RAG memory retrieval
            execution_result: Result/error from last code execution
            current_step: Current step number
            is_first_turn: Whether this is the first turn

        Returns:
            List of messages ready for LLM invocation
        """
        messages = []

        # ============================================================
        # 1. STATIC INFORMATION (from YAML)
        # ============================================================
        messages.append(SystemMessage(content=self._system_task))
        messages.append(HumanMessage(content=f"# Your Goal:\n{self._human_task}"))

        # ============================================================
        # 2. RAG MEMORIES (dynamic)
        # ============================================================
        if memory_text and memory_text.strip():
            memory_section = self._format_memory_section(memory_text)
            messages.append(SystemMessage(content=memory_section))

        # ============================================================
        # 3. CURRENT OBSERVATION (dead reckoning + vision)
        # ============================================================
        current_obs = self._format_current_observation_no_metadata(
            agent_position=agent_position,
            agent_rotation=agent_rotation,
            detected_objects=detected_objects,
            action_success=action_success,
            error_message=error_message,
            execution_result=execution_result,
            current_step=current_step,
            is_first_turn=is_first_turn
        )

        # Combine text observation with screenshot
        messages.append(HumanMessage(content=[
            {"type": "text", "text": current_obs},
            {"type": "image_url", "image_url": {"url": encoded_image}}
        ]))

        return messages

    def _format_current_observation_no_metadata(self,
                                                  agent_position: Dict[str, float],
                                                  agent_rotation: Dict[str, float],
                                                  detected_objects: List["RelativeCoordinate"],
                                                  action_success: bool,
                                                  error_message: str,
                                                  execution_result: Optional[str],
                                                  current_step: int,
                                                  is_first_turn: bool) -> str:
        """Format current observation for no-metadata mode.

        Uses Dead Reckoning position (not simulator metadata).

        Args:
            agent_position: Dead reckoning estimated position
            agent_rotation: Dead reckoning estimated rotation
            detected_objects: Vision-detected objects with estimated positions
            action_success: Whether last action succeeded
            error_message: Error message if action failed
            execution_result: Last execution result
            current_step: Current step number
            is_first_turn: Whether this is the first turn

        Returns:
            Formatted observation string
        """
        sections = []

        # Header
        sections.append(f"# Current Observation (Step {current_step}) [VISION + DEAD RECKONING]")
        sections.append("(Position from movement tracking, objects from vision.)\n")

        # Last Action Result (except on first turn)
        if not is_first_turn:
            if action_success:
                action_status = "SUCCESS"
            else:
                action_status = f"FAILED - {error_message}"

            sections.append(f"## Last Action Result: {action_status}")

            if execution_result:
                sections.append(f"Execution Output: {execution_result}")

            sections.append("")

        # Agent State (Dead Reckoning estimated position)
        facing = self._get_facing_direction(agent_rotation['y'])
        sections.append(f"## Agent State (Dead Reckoning):")
        sections.append(f"- Estimated Position: ({agent_position['x']:.2f}, {agent_position['z']:.2f})")
        sections.append(f"- Facing: {facing} ({agent_rotation['y']:.0f}°)")
        sections.append("")

        # Detected Objects (from vision)
        sections.append("## Detected Objects (vision-estimated positions):")
        if detected_objects:
            for coord in detected_objects:
                direction = "right" if coord.x_local >= 0 else "left"
                sections.append(
                    f"- {coord.object_type}: ~{abs(coord.x_local):.1f}m {direction}, "
                    f"~{coord.z_local:.1f}m ahead"
                )
        else:
            sections.append("- (No objects detected in current view)")
        sections.append("")

        # Instructions
        sections.append("## Instructions:")
        sections.append("Based on your Goal, Memories, and what you see in the screenshot,")
        sections.append("generate Python code for your next action.")
        sections.append("")
        sections.append("IMPORTANT: You must identify objects visually in the screenshot.")
        sections.append("Use the estimated positions above as hints, but verify visually.")
        sections.append("")
        sections.append("Below is a screenshot from the simulator:")

        return "\n".join(sections)

    def _get_facing_direction(self, rotation_y: float) -> str:
        """Convert Y rotation to cardinal direction."""
        rotation_y = rotation_y % 360

        if rotation_y < 45 or rotation_y >= 315:
            return "North"
        elif rotation_y < 135:
            return "East"
        elif rotation_y < 225:
            return "South"
        else:
            return "West"

    def format_visible_objects(self, env_state: EnvironmentState) -> str:
        """Format visible objects as a standalone string.

        Args:
            env_state: Environment state containing visible objects

        Returns:
            Formatted visible objects string
        """
        if not env_state.visible_objects:
            return "Visible objects: None"

        lines = [f"- type=\"{obj.object_type}\" id=\"{obj.object_id}\""
                 for obj in env_state.visible_objects]
        return "Visible objects:\n" + "\n".join(lines)


class PromptTemplate:
    """Pre-defined prompt templates for various situations."""

    @staticmethod
    def exploration_hint(missing_objects: List[str]) -> str:
        """Generate a hint for exploring to find missing objects.

        Args:
            missing_objects: List of object types that need to be found

        Returns:
            Exploration hint string
        """
        if not missing_objects:
            return ""

        return (
            f"\n[HINT: You need to find: {', '.join(missing_objects)}. "
            f"Try rotating or moving to explore the environment.]\n"
        )

    @staticmethod
    def navigation_hint(target_object: str, last_known_position: Optional[Dict[str, float]]) -> str:
        """Generate a navigation hint to reach a target object.

        Args:
            target_object: The object to navigate to
            last_known_position: Last known position of the object (optional)

        Returns:
            Navigation hint string
        """
        if last_known_position:
            return (
                f"\n[HINT: {target_object} was last seen at position "
                f"({last_known_position['x']:.2f}, {last_known_position['z']:.2f}). "
                f"Navigate towards that location.]\n"
            )
        else:
            return f"\n[HINT: {target_object} location unknown. Explore to find it.]\n"

    @staticmethod
    def error_recovery_hint(failed_action: str, error_message: str) -> str:
        """Generate a hint for recovering from an error.

        Args:
            failed_action: The action that failed
            error_message: The error message

        Returns:
            Error recovery hint string
        """
        hints = []

        if "not visible" in error_message.lower():
            hints.append("Move closer to the object or rotate to see it.")
        elif "out of range" in error_message.lower() or "too far" in error_message.lower():
            hints.append("Move closer to the object (within 1.5 meters).")
        elif "already holding" in error_message.lower():
            hints.append("Put down the object you're holding first.")
        elif "nothing" in error_message.lower() and "holding" in error_message.lower():
            hints.append("Pick up an object first before trying to put it down.")
        elif "blocked" in error_message.lower() or "collision" in error_message.lower():
            hints.append("Try a different direction or path.")
        elif "closed" in error_message.lower():
            hints.append("Open the container first before accessing its contents.")

        if hints:
            return f"\n[RECOVERY HINT: {' '.join(hints)}]\n"
        return ""
