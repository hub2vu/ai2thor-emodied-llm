"""MemoryWriter - Writes observations and facts to the memory store.

This module processes environment states and extracts salient information
to be stored in the vector database for later retrieval.

Enhanced features:
- Frame/image storage with frame_id linking
- Automatic relation extraction from parentReceptacles
- Observation signature hash for intelligent deduplication
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import hashlib
import math
import os
import base64
from pathlib import Path

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
    is_toggled: Optional[bool] = None
    parent_receptacles: Optional[List[str]] = None


class MemoryWriter:
    """Writes environment observations to the memory store.

    Responsible for:
    - Processing EnvironmentState and extracting salient facts
    - Deduplicating information using observation signature hash
    - Managing object state tracking with relation extraction
    - Creating keyframe observations with linked frame images
    - Automatic relation extraction from parentReceptacles
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

    def __init__(self, memory_store: MemoryStore,
                 image_storage_dir: Optional[str] = None,
                 store_images: bool = True):
        """Initialize the memory writer.

        Args:
            memory_store: The memory store to write to
            image_storage_dir: Directory to store frame images (default: ./frames)
            store_images: Whether to store frame images to disk
        """
        self._store = memory_store
        self._current_step = 0
        self._store_images = store_images

        # Setup image storage directory
        if image_storage_dir:
            self._image_dir = Path(image_storage_dir)
        else:
            self._image_dir = Path("./frames")

        if self._store_images:
            self._image_dir.mkdir(parents=True, exist_ok=True)

        # Track object states to detect changes
        self._object_states: Dict[str, ObjectState] = {}

        # Track agent positions for keyframe detection
        self._last_agent_position: Optional[Dict[str, float]] = None
        self._last_agent_rotation: Optional[Dict[str, float]] = None
        self._visited_positions: Set[tuple] = set()

        # Track last action for context
        self._last_action_success: Optional[bool] = None
        self._last_action_error: Optional[str] = None

        # Observation signature tracking for deduplication
        self._last_observation_signature: Optional[str] = None

        # Frame ID counter
        self._frame_counter = 0

    @property
    def current_step(self) -> int:
        """Get the current step number."""
        return self._current_step

    def _compute_observation_signature(self, env_state: EnvironmentState) -> str:
        """Compute a signature hash for the current observation.

        The signature is based on:
        - Visible object IDs (sorted)
        - Object states (open/toggled)
        - Object positions (quantized)
        - Agent position (quantized)

        Args:
            env_state: The current environment state

        Returns:
            A hex string hash representing the observation
        """
        signature_parts = []

        # Agent position (quantized to 0.1m grid)
        agent_pos = env_state.agent_position
        signature_parts.append(f"agent:{round(agent_pos['x'], 1)},{round(agent_pos['z'], 1)}")

        # Agent rotation (quantized to 45 degrees)
        rot_quantized = round(env_state.agent_rotation['y'] / 45) * 45
        signature_parts.append(f"rot:{rot_quantized}")

        # Visible objects and their states
        obj_signatures = []
        for obj in env_state.visible_objects:
            if obj.object_type in self.IGNORE_OBJECTS:
                continue

            # Include object ID, position (quantized), and state
            pos_q = f"{round(obj.position['x'], 1)},{round(obj.position['z'], 1)}"
            state = f"o{int(obj.is_open)}t{int(obj.is_toggled)}"
            parent = ",".join(sorted(obj.parent_receptacles)) if obj.parent_receptacles else "none"
            obj_signatures.append(f"{obj.object_id}@{pos_q}:{state}:{parent}")

        # Sort for consistency
        obj_signatures.sort()
        signature_parts.extend(obj_signatures)

        # Create hash
        signature_string = "|".join(signature_parts)
        return hashlib.md5(signature_string.encode()).hexdigest()

    def _is_observation_duplicate(self, env_state: EnvironmentState) -> bool:
        """Check if current observation is a duplicate using signature hash.

        Args:
            env_state: The current environment state

        Returns:
            True if this observation is essentially the same as the last one
        """
        current_signature = self._compute_observation_signature(env_state)

        if current_signature == self._last_observation_signature:
            return True

        self._last_observation_signature = current_signature
        return False

    def _save_frame_image(self, image_data_url: str, frame_id: str) -> Optional[str]:
        """Save a frame image to disk.

        Args:
            image_data_url: Base64 data URL of the image
            frame_id: Unique identifier for this frame

        Returns:
            Path to saved image file, or None if storage is disabled
        """
        if not self._store_images:
            return None

        try:
            # Extract base64 data from data URL
            if "base64," in image_data_url:
                base64_data = image_data_url.split("base64,")[1]
            else:
                base64_data = image_data_url

            # Decode and save
            image_bytes = base64.b64decode(base64_data)
            image_path = self._image_dir / f"{frame_id}.jpg"

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            return str(image_path)
        except Exception as e:
            print(f"Warning: Failed to save frame image: {e}")
            return None

    def _generate_frame_id(self) -> str:
        """Generate a unique frame ID.

        Returns:
            A unique frame identifier string
        """
        self._frame_counter += 1
        return f"frame_{self._current_step:04d}_{self._frame_counter:04d}"

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

        # Check for duplicate observation (using signature hash)
        is_duplicate = self._is_observation_duplicate(env_state)

        # 1. Process action result (if action failed, this is important to remember)
        if action_taken and not env_state.last_action_success:
            action_doc = self._create_action_memory(
                action_taken, env_state.last_action_success,
                env_state.error_message
            )
            doc_id = self._store.add(action_doc)
            written_ids.append(doc_id)

        # 2. Check for significant position change (keyframe observation)
        # Skip if duplicate observation
        if not is_duplicate and self._is_significant_position_change(env_state):
            # Generate frame ID and save image
            frame_id = self._generate_frame_id()
            image_path = self._save_frame_image(env_state.agent_camera_view, frame_id)

            obs_doc = self._create_observation_memory(env_state, frame_id, image_path)
            doc_id = self._store.add(obs_doc)
            written_ids.append(doc_id)
            self._update_agent_tracking(env_state)

        # 3. Process visible objects and extract relations
        if not is_duplicate:
            object_docs, relation_docs = self._process_visible_objects(env_state)

            if object_docs:
                ids = self._store.add_batch(object_docs)
                written_ids.extend(ids)

            if relation_docs:
                ids = self._store.add_batch(relation_docs)
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

    def _create_observation_memory(self, env_state: EnvironmentState,
                                     frame_id: str,
                                     image_path: Optional[str] = None) -> MemoryDocument:
        """Create a keyframe observation memory with linked frame.

        Args:
            env_state: The environment state
            frame_id: Unique identifier for this frame
            image_path: Path to the saved frame image

        Returns:
            MemoryDocument for the observation
        """
        pos = env_state.agent_position
        rot = env_state.agent_rotation

        # Determine facing direction
        facing = self._get_facing_direction(rot["y"])

        # Get summary of visible objects
        visible_types = [obj.object_type for obj in env_state.visible_objects
                         if obj.object_type not in self.IGNORE_OBJECTS][:5]

        content = (
            f"Agent at position ({pos['x']:.2f}, {pos['z']:.2f}) "
            f"facing {facing}. "
            f"Visible objects: {', '.join(visible_types)}"
        )

        metadata = {
            "agent_x": pos["x"],
            "agent_z": pos["z"],
            "agent_rotation": rot["y"],
            "facing": facing,
            "visible_count": len(env_state.visible_objects),
            "frame_id": frame_id,
            "observation_signature": self._last_observation_signature
        }

        if image_path:
            metadata["image_path"] = image_path

        return MemoryDocument(
            content=content,
            memory_type=MemoryType.OBSERVATION,
            step=self._current_step,
            metadata=metadata
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

    def _process_visible_objects(self, env_state: EnvironmentState) -> Tuple[List[MemoryDocument], List[MemoryDocument]]:
        """Process visible objects and create/update memories.

        Also extracts relations from parentReceptacles.

        Args:
            env_state: The environment state

        Returns:
            Tuple of (object documents, relation documents)
        """
        object_docs = []
        relation_docs = []

        for obj in env_state.visible_objects:
            # Skip ignored objects
            if obj.object_type in self.IGNORE_OBJECTS:
                continue

            # Check if object state has changed
            state_changed, changes = self._detect_object_changes(obj)

            if state_changed:
                doc = self._create_object_memory(obj, env_state, changes)
                object_docs.append(doc)
                self._update_object_state(obj)

                # Extract relations from parentReceptacles
                if obj.parent_receptacles:
                    for receptacle_id in obj.parent_receptacles:
                        receptacle_type = receptacle_id.split("|")[0] if "|" in receptacle_id else receptacle_id
                        relation_doc = self._create_relation_memory(
                            subject_id=obj.object_id,
                            subject_type=obj.object_type,
                            relation="IN" if receptacle_type in {"Fridge", "Cabinet", "Drawer", "Microwave"} else "ON",
                            target_id=receptacle_id,
                            target_type=receptacle_type
                        )
                        relation_docs.append(relation_doc)

        return object_docs, relation_docs

    def _detect_object_changes(self, obj: SimulatorObject) -> Tuple[bool, List[str]]:
        """Detect what has changed about an object.

        Args:
            obj: The simulator object

        Returns:
            Tuple of (has_changed, list_of_changes)
        """
        changes = []

        if obj.object_id not in self._object_states:
            changes.append("first_seen")
            return True, changes

        old_state = self._object_states[obj.object_id]

        # Check position change
        dx = obj.position["x"] - old_state.position["x"]
        dy = obj.position["y"] - old_state.position["y"]
        dz = obj.position["z"] - old_state.position["z"]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        if distance > 0.1:  # Object moved more than 10cm
            changes.append("position_changed")

        # Check state changes
        if old_state.is_open != obj.is_open:
            changes.append("open_state_changed")

        if old_state.is_toggled != obj.is_toggled:
            changes.append("toggle_state_changed")

        # Check parent receptacle changes
        old_parents = set(old_state.parent_receptacles or [])
        new_parents = set(obj.parent_receptacles or [])
        if old_parents != new_parents:
            changes.append("container_changed")

        return len(changes) > 0, changes

    def _update_object_state(self, obj: SimulatorObject) -> None:
        """Update the tracked state of an object."""
        self._object_states[obj.object_id] = ObjectState(
            object_id=obj.object_id,
            object_type=obj.object_type,
            position=obj.position.copy(),
            last_seen_step=self._current_step,
            is_open=obj.is_open,
            is_toggled=obj.is_toggled,
            parent_receptacles=list(obj.parent_receptacles) if obj.parent_receptacles else []
        )

    def _create_object_memory(self, obj: SimulatorObject,
                               env_state: EnvironmentState,
                               changes: List[str]) -> MemoryDocument:
        """Create a memory document for an object.

        Args:
            obj: The simulator object
            env_state: The environment state
            changes: List of detected changes

        Returns:
            MemoryDocument for the object
        """
        # Determine location context
        location = self._describe_location(obj.position, env_state.agent_position)

        # Build state description
        state_parts = []
        if obj.is_open:
            state_parts.append("open")
        if obj.is_toggled:
            state_parts.append("on")
        if obj.parent_receptacles:
            containers = [r.split("|")[0] for r in obj.parent_receptacles]
            state_parts.append(f"in/on {', '.join(containers)}")

        state_str = f" ({', '.join(state_parts)})" if state_parts else ""

        content = (
            f"Object '{obj.object_type}' (id: {obj.object_id}) "
            f"is at {location}{state_str}."
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
                "position_z": obj.position["z"],
                "is_open": obj.is_open,
                "is_toggled": obj.is_toggled,
                "is_pickupable": obj.is_pickupable,
                "is_receptacle": obj.is_receptacle,
                # ChromaDB only supports str/int/float/bool - convert lists to comma-separated strings
                "parent_receptacles": ",".join(obj.parent_receptacles) if obj.parent_receptacles else "",
                "changes": ",".join(changes) if changes else ""
            }
        )

    def _create_relation_memory(self, subject_id: str, subject_type: str,
                                 relation: str, target_id: str,
                                 target_type: str) -> MemoryDocument:
        """Create a relation memory document.

        Args:
            subject_id: ID of the subject object
            subject_type: Type of the subject object
            relation: The relationship (ON, IN, CONTAINS, etc.)
            target_id: ID of the target object
            target_type: Type of the target object

        Returns:
            MemoryDocument for the relation
        """
        content = f"{subject_type} (id: {subject_id}) is {relation} {target_type} (id: {target_id})."

        return MemoryDocument(
            content=content,
            memory_type=MemoryType.RELATION,
            step=self._current_step,
            metadata={
                "subject_id": subject_id,
                "subject_type": subject_type,
                "relation": relation,
                "target_id": target_id,
                "target_type": target_type
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

    def record_observation(self, agent_position: Dict[str, float],
                            agent_rotation: Dict[str, float],
                            action_success: bool = True,
                            error_message: str = "") -> Optional[str]:
        """Record a basic observation without using visible_objects metadata.

        This method is used in no-metadata mode where object detection
        is handled separately via VLM.

        Args:
            agent_position: Agent's current position
            agent_rotation: Agent's current rotation
            action_success: Whether the last action succeeded
            error_message: Error message if action failed

        Returns:
            Document ID if observation was recorded, None if skipped
        """
        self._current_step += 1

        # Check for significant position change
        if self._last_agent_position is not None:
            dx = agent_position["x"] - self._last_agent_position["x"]
            dz = agent_position["z"] - self._last_agent_position["z"]
            distance = math.sqrt(dx * dx + dz * dz)

            rotation_diff = 0
            if self._last_agent_rotation:
                rotation_diff = abs(agent_rotation["y"] - self._last_agent_rotation["y"])
                rotation_diff = min(rotation_diff, 360 - rotation_diff)

            # Skip if no significant movement
            if distance < self.POSITION_CHANGE_THRESHOLD and rotation_diff < self.ROTATION_CHANGE_THRESHOLD:
                # Still record failed actions
                if not action_success:
                    return self._record_failed_action(error_message)
                return None

        # Record significant position change
        facing = self._get_facing_direction(agent_rotation["y"])
        content = (
            f"Agent at position ({agent_position['x']:.2f}, {agent_position['z']:.2f}) "
            f"facing {facing}."
        )

        doc = MemoryDocument(
            content=content,
            memory_type=MemoryType.OBSERVATION,
            step=self._current_step,
            metadata={
                "agent_x": agent_position["x"],
                "agent_z": agent_position["z"],
                "agent_rotation": agent_rotation["y"],
                "facing": facing,
                "coordinate_source": "vision_mode"  # Mark as vision mode observation
            }
        )

        doc_id = self._store.add(doc)

        # Update tracking
        self._last_agent_position = agent_position.copy()
        self._last_agent_rotation = agent_rotation.copy()

        # Track visited positions
        grid_pos = (round(agent_position["x"], 1), round(agent_position["z"], 1))
        self._visited_positions.add(grid_pos)

        return doc_id

    def _record_failed_action(self, error_message: str) -> str:
        """Record a failed action in no-metadata mode.

        Args:
            error_message: The error message

        Returns:
            The document ID
        """
        content = f"Last action FAILED: {error_message}"

        doc = MemoryDocument(
            content=content,
            memory_type=MemoryType.ACTION,
            step=self._current_step,
            metadata={
                "success": False,
                "error": error_message
            }
        )

        return self._store.add(doc)

    def record_vision_based_object(self, object_type: str,
                                     x_local: float, z_local: float,
                                     confidence: float = 1.0,
                                     is_fixed: bool = False) -> str:
        """Record an object position estimated from vision (no metadata).

        This method stores object positions derived purely from visual
        observation using the VisionCoordinateEstimator.

        Args:
            object_type: The type of object (e.g., "Fridge", "Sink")
            x_local: Relative X coordinate (right=+, left=-)
            z_local: Relative Z coordinate (depth/distance, front=+)
            confidence: Confidence score of the detection (0-1)
            is_fixed: Whether this is a fixed/immovable object

        Returns:
            The document ID
        """
        direction = "right" if x_local >= 0 else "left"
        content = (
            f"Object '{object_type}' is approximately {abs(x_local):.1f}m "
            f"to your {direction} and {z_local:.1f}m ahead."
        )

        doc = MemoryDocument(
            content=content,
            memory_type=MemoryType.OBJECT,
            step=self._current_step,
            metadata={
                "object_type": object_type,
                "x_local": x_local,
                "z_local": z_local,
                "coordinate_source": "vision",  # Mark as vision-derived
                "confidence": confidence,
                "is_fixed": is_fixed,
                "is_pickupable": not is_fixed  # Fixed objects are not pickupable
            }
        )

        return self._store.add(doc)

    def record_vision_observation(self, detected_objects: List[Dict[str, Any]],
                                   frame_id: Optional[str] = None) -> List[str]:
        """Record multiple objects detected via vision in a single observation.

        Args:
            detected_objects: List of dicts with keys:
                - object_type: str
                - x_local: float
                - z_local: float
                - confidence: float (optional)
                - is_fixed: bool (optional)
            frame_id: Optional frame ID to link to image

        Returns:
            List of document IDs
        """
        doc_ids = []

        for obj in detected_objects:
            doc_id = self.record_vision_based_object(
                object_type=obj["object_type"],
                x_local=obj["x_local"],
                z_local=obj["z_local"],
                confidence=obj.get("confidence", 1.0),
                is_fixed=obj.get("is_fixed", False)
            )
            doc_ids.append(doc_id)

        return doc_ids

    def get_visited_positions(self) -> Set[tuple]:
        """Get the set of visited grid positions."""
        return self._visited_positions.copy()

    def get_observation_signature(self) -> Optional[str]:
        """Get the current observation signature hash."""
        return self._last_observation_signature

    def reset(self) -> None:
        """Reset the writer state for a new episode."""
        self._current_step = 0
        self._object_states.clear()
        self._last_agent_position = None
        self._last_agent_rotation = None
        self._visited_positions.clear()
        self._last_action_success = None
        self._last_action_error = None
        self._last_observation_signature = None
        self._frame_counter = 0
        self._store.clear()
