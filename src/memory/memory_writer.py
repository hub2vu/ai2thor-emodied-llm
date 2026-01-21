"""MemoryWriter - Writes observations and facts to the memory store.

This module processes environment states and extracts salient information
to be stored in the vector database for later retrieval.
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import math

from src.simulator import EnvironmentState, SimulatorObject
from src.memory.memory_store import MemoryStore, MemoryDocument, MemoryType


@dataclass
class ObjectState:
    """Tracks the state of an object across time."""
    object_id: str
    object_type: str
    position: Dict[str, float]
    last_seen_step: int
    is_held: bool = False
    is_open: Optional[bool] = None
    is_on: Optional[bool] = None
    parent_receptacle: Optional[str] = None


class MemoryWriter:
    """Writes environment observations to the memory store.

    Responsible for:
    - Processing EnvironmentState and extracting salient facts
    - Deduplicating information (avoiding repeated writes of unchanged facts)
    - Managing object state tracking
    - Creating keyframe observations for navigation history
    """

    # Objects that don't need detailed tracking
    IGNORE_OBJECTS = {
        "Floor", "Wall", "Ceiling", "Window", "Door",
        "LightSwitch", "Painting", "Curtains"
    }

    # Distance threshold for position change (in meters)
    POSITION_CHANGE_THRESHOLD = 0.3

    # Rotation threshold for significant turn (in degrees)
    ROTATION_CHANGE_THRESHOLD = 45

    def __init__(self, memory_store: MemoryStore):
        """Initialize the memory writer.

        Args:
            memory_store: The memory store to write to
        """
        self._store = memory_store
        self._current_step = 0

        # Track object states to detect changes
        self._object_states: Dict[str, ObjectState] = {}

        # Track agent positions for keyframe detection
        self._last_agent_position: Optional[Dict[str, float]] = None
        self._last_agent_rotation: Optional[Dict[str, float]] = None
        self._visited_positions: Set[tuple] = set()

        # Track last action for context
        self._last_action_success: Optional[bool] = None
        self._last_action_error: Optional[str] = None

    @property
    def current_step(self) -> int:
        """Get the current step number."""
        return self._current_step

    def process_environment_state(self,
                                   env_state: EnvironmentState,
                                   action_taken: Optional[str] = None,
                                   execution_result: Optional[str] = None) -> List[str]:
        """Process an environment state and write relevant memories.

        Args:
            env_state: The current environment state
            action_taken: The action that was taken (if any)
            execution_result: The result of code execution (if any)

        Returns:
            List of document IDs that were written
        """
        self._current_step += 1
        written_ids = []

        # 1. Process action result (if action failed, this is important to remember)
        if action_taken and not env_state.last_action_success:
            action_doc = self._create_action_memory(
                action_taken, env_state.last_action_success,
                env_state.error_message
            )
            doc_id = self._store.add(action_doc)
            written_ids.append(doc_id)

        # 2. Check for significant position change (keyframe observation)
        if self._is_significant_position_change(env_state):
            obs_doc = self._create_observation_memory(env_state)
            doc_id = self._store.add(obs_doc)
            written_ids.append(doc_id)
            self._update_agent_tracking(env_state)

        # 3. Process visible objects
        object_docs = self._process_visible_objects(env_state)
        if object_docs:
            ids = self._store.add_batch(object_docs)
            written_ids.extend(ids)

        # Update tracking
        self._last_action_success = env_state.last_action_success
        self._last_action_error = env_state.error_message

        return written_ids

    def _is_significant_position_change(self, env_state: EnvironmentState) -> bool:
        """Check if agent position has changed significantly."""
        if self._last_agent_position is None:
            return True

        # Calculate distance moved
        dx = env_state.agent_position["x"] - self._last_agent_position["x"]
        dz = env_state.agent_position["z"] - self._last_agent_position["z"]
        distance = math.sqrt(dx * dx + dz * dz)

        if distance >= self.POSITION_CHANGE_THRESHOLD:
            return True

        # Check rotation change
        if self._last_agent_rotation:
            rotation_diff = abs(
                env_state.agent_rotation["y"] - self._last_agent_rotation["y"]
            )
            # Handle wraparound
            rotation_diff = min(rotation_diff, 360 - rotation_diff)
            if rotation_diff >= self.ROTATION_CHANGE_THRESHOLD:
                return True

        return False

    def _update_agent_tracking(self, env_state: EnvironmentState) -> None:
        """Update agent position/rotation tracking."""
        self._last_agent_position = env_state.agent_position.copy()
        self._last_agent_rotation = env_state.agent_rotation.copy()

        # Track visited positions (rounded to grid)
        grid_pos = (
            round(env_state.agent_position["x"], 1),
            round(env_state.agent_position["z"], 1)
        )
        self._visited_positions.add(grid_pos)

    def _create_observation_memory(self, env_state: EnvironmentState) -> MemoryDocument:
        """Create a keyframe observation memory."""
        pos = env_state.agent_position
        rot = env_state.agent_rotation

        # Determine facing direction
        facing = self._get_facing_direction(rot["y"])

        content = (
            f"Agent at position ({pos['x']:.2f}, {pos['z']:.2f}) "
            f"facing {facing}. "
            f"Visible objects: {', '.join(obj.object_type for obj in env_state.visible_objects[:5])}"
        )

        return MemoryDocument(
            content=content,
            memory_type=MemoryType.OBSERVATION,
            step=self._current_step,
            metadata={
                "agent_x": pos["x"],
                "agent_z": pos["z"],
                "agent_rotation": rot["y"],
                "facing": facing,
                "visible_count": len(env_state.visible_objects)
            }
        )

    def _create_action_memory(self, action: str, success: bool, error: str) -> MemoryDocument:
        """Create a memory for an action result (especially failures)."""
        if success:
            content = f"Action '{action}' succeeded."
        else:
            content = f"Action '{action}' FAILED: {error}"

        return MemoryDocument(
            content=content,
            memory_type=MemoryType.ACTION,
            step=self._current_step,
            metadata={
                "action": action,
                "success": success,
                "error": error if not success else ""
            }
        )

    def _process_visible_objects(self, env_state: EnvironmentState) -> List[MemoryDocument]:
        """Process visible objects and create/update memories."""
        docs = []

        for obj in env_state.visible_objects:
            # Skip ignored objects
            if obj.object_type in self.IGNORE_OBJECTS:
                continue

            # Check if object state has changed
            if self._has_object_state_changed(obj):
                doc = self._create_object_memory(obj, env_state)
                docs.append(doc)
                self._update_object_state(obj)

        return docs

    def _has_object_state_changed(self, obj: SimulatorObject) -> bool:
        """Check if an object's state has changed significantly."""
        if obj.object_id not in self._object_states:
            return True

        old_state = self._object_states[obj.object_id]

        # Check position change
        dx = obj.position["x"] - old_state.position["x"]
        dy = obj.position["y"] - old_state.position["y"]
        dz = obj.position["z"] - old_state.position["z"]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        if distance > 0.1:  # Object moved more than 10cm
            return True

        return False

    def _update_object_state(self, obj: SimulatorObject) -> None:
        """Update the tracked state of an object."""
        self._object_states[obj.object_id] = ObjectState(
            object_id=obj.object_id,
            object_type=obj.object_type,
            position=obj.position.copy(),
            last_seen_step=self._current_step
        )

    def _create_object_memory(self, obj: SimulatorObject,
                               env_state: EnvironmentState) -> MemoryDocument:
        """Create a memory document for an object."""
        # Determine location context
        location = self._describe_location(obj.position, env_state.agent_position)

        content = (
            f"Object '{obj.object_type}' (id: {obj.object_id}) "
            f"is at {location}."
        )

        return MemoryDocument(
            content=content,
            memory_type=MemoryType.OBJECT,
            step=self._current_step,
            metadata={
                "object_id": obj.object_id,
                "object_type": obj.object_type,
                "position_x": obj.position["x"],
                "position_y": obj.position["y"],
                "position_z": obj.position["z"]
            }
        )

    def _describe_location(self, obj_pos: Dict[str, float],
                           agent_pos: Dict[str, float]) -> str:
        """Describe object location relative to agent."""
        dx = obj_pos["x"] - agent_pos["x"]
        dz = obj_pos["z"] - agent_pos["z"]
        distance = math.sqrt(dx * dx + dz * dz)

        return f"position ({obj_pos['x']:.2f}, {obj_pos['y']:.2f}, {obj_pos['z']:.2f}), {distance:.2f}m from agent"

    def _get_facing_direction(self, rotation_y: float) -> str:
        """Convert Y rotation to cardinal direction."""
        # Normalize to 0-360
        rotation_y = rotation_y % 360

        if rotation_y < 45 or rotation_y >= 315:
            return "North"
        elif rotation_y < 135:
            return "East"
        elif rotation_y < 225:
            return "South"
        else:
            return "West"

    def record_relation(self, subject: str, relation: str, target: str,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """Record a relationship between objects.

        Args:
            subject: The subject object (e.g., "Pot")
            relation: The relationship (e.g., "ON", "IN", "CONTAINS")
            target: The target object (e.g., "StoveBurner")
            metadata: Additional metadata

        Returns:
            The document ID
        """
        content = f"{subject} is {relation} {target}."

        doc = MemoryDocument(
            content=content,
            memory_type=MemoryType.RELATION,
            step=self._current_step,
            metadata={
                "subject": subject,
                "relation": relation,
                "target": target,
                **(metadata or {})
            }
        )

        return self._store.add(doc)

    def record_held_object(self, object_id: str, object_type: str) -> str:
        """Record that the agent is holding an object.

        Args:
            object_id: The held object's ID
            object_type: The held object's type

        Returns:
            The document ID
        """
        content = f"Agent is holding {object_type} (id: {object_id})."

        doc = MemoryDocument(
            content=content,
            memory_type=MemoryType.RELATION,
            step=self._current_step,
            metadata={
                "subject": "Agent",
                "relation": "HOLDING",
                "target": object_id,
                "object_type": object_type
            }
        )

        return self._store.add(doc)

    def record_object_state_change(self, object_id: str, object_type: str,
                                    state: str, value: bool) -> str:
        """Record an object state change (open/closed, on/off).

        Args:
            object_id: The object's ID
            object_type: The object's type
            state: The state name (e.g., "open", "on")
            value: The new state value

        Returns:
            The document ID
        """
        state_str = state if value else f"not {state}"
        content = f"{object_type} (id: {object_id}) is now {state_str}."

        doc = MemoryDocument(
            content=content,
            memory_type=MemoryType.OBJECT,
            step=self._current_step,
            metadata={
                "object_id": object_id,
                "object_type": object_type,
                "state": state,
                "value": value
            }
        )

        return self._store.add(doc)

    def get_visited_positions(self) -> Set[tuple]:
        """Get the set of visited grid positions."""
        return self._visited_positions.copy()

    def reset(self) -> None:
        """Reset the writer state for a new episode."""
        self._current_step = 0
        self._object_states.clear()
        self._last_agent_position = None
        self._last_agent_rotation = None
        self._visited_positions.clear()
        self._last_action_success = None
        self._last_action_error = None
        self._store.clear()
