"""MemoryRetriever - Retrieves relevant memories for context injection.

This module provides intelligent retrieval of memories based on the current
task and situation, with recency-weighted search and automatic query generation.
"""

from typing import List, Dict, Any, Optional, Set
import re

from src.simulator import EnvironmentState, SimulatorObject
from src.memory.memory_store import MemoryStore, MemoryType


class MemoryRetriever:
    """Retrieves relevant memories for context injection.

    Responsible for:
    - Generating queries based on task description and current view
    - Retrieving object locations (with recency weighting)
    - Retrieving navigation history
    - Identifying missing objects needed for task
    """

    # Weight for recency in scoring (0-1, higher = more recency preference)
    RECENCY_WEIGHT = 0.8

    # Maximum number of memories to retrieve per category
    MAX_OBJECT_MEMORIES = 10
    MAX_OBSERVATION_MEMORIES = 5
    MAX_ACTION_MEMORIES = 5
    MAX_RELATION_MEMORIES = 5

    def __init__(self, memory_store: MemoryStore):
        """Initialize the memory retriever.

        Args:
            memory_store: The memory store to query
        """
        self._store = memory_store

    def retrieve_relevant_context(self,
                                   task_description: str,
                                   env_state: EnvironmentState,
                                   current_step: int) -> Dict[str, Any]:
        """Retrieve all relevant context for the current situation.

        Args:
            task_description: The task goal description
            env_state: Current environment state
            current_step: The current step number

        Returns:
            Dictionary containing categorized memories
        """
        # Extract task-relevant nouns (object types mentioned in task)
        task_objects = self._extract_task_objects(task_description)

        # Get currently visible object types
        visible_types = {obj.object_type for obj in env_state.visible_objects}
        visible_ids = {obj.object_id for obj in env_state.visible_objects}

        # Find objects mentioned in task but not currently visible
        missing_objects = task_objects - visible_types

        # Retrieve memories
        context = {
            "object_memories": [],
            "observation_memories": [],
            "action_memories": [],
            "relation_memories": [],
            "missing_objects": list(missing_objects)
        }

        # 1. Retrieve object memories (prioritize missing objects)
        context["object_memories"] = self._retrieve_object_memories(
            task_objects, missing_objects, current_step
        )

        # 2. Retrieve recent failed actions (important for avoiding loops)
        context["action_memories"] = self._retrieve_failed_actions(current_step)

        # 3. Retrieve relevant relations
        context["relation_memories"] = self._retrieve_relations(
            task_objects, current_step
        )

        # 4. Retrieve navigation observations (for spatial awareness)
        context["observation_memories"] = self._retrieve_observations(
            current_step
        )

        return context

    def retrieve_relevant_context_no_metadata(self,
                                               task_description: str,
                                               agent_position: Dict[str, float],
                                               current_step: int) -> Dict[str, Any]:
        """Retrieve context for no-metadata mode (Dead Reckoning + Vision).

        In no-metadata mode, we don't have access to env_state.visible_objects,
        so we can't determine what's currently visible vs missing.
        Instead, we retrieve all task-relevant memories.

        Args:
            task_description: The task goal description
            agent_position: Dead reckoning estimated position
            current_step: The current step number

        Returns:
            Dictionary containing categorized memories
        """
        # Extract task-relevant nouns (object types mentioned in task)
        task_objects = self._extract_task_objects(task_description)

        # In no-metadata mode, we don't know what's visible
        # So we treat all task objects as potentially needing memory lookup
        context = {
            "object_memories": [],
            "observation_memories": [],
            "action_memories": [],
            "relation_memories": [],
            "missing_objects": []  # Can't determine without metadata
        }

        # 1. Retrieve object memories for all task objects
        context["object_memories"] = self._retrieve_object_memories_no_metadata(
            task_objects, current_step
        )

        # 2. Retrieve recent failed actions
        context["action_memories"] = self._retrieve_failed_actions(current_step)

        # 3. Retrieve relevant relations
        context["relation_memories"] = self._retrieve_relations(
            task_objects, current_step
        )

        # 4. Retrieve navigation observations near current position
        context["observation_memories"] = self._retrieve_observations_near_position(
            agent_position, current_step
        )

        return context

    def _retrieve_object_memories_no_metadata(self,
                                               task_objects: Set[str],
                                               current_step: int) -> List[Dict[str, Any]]:
        """Retrieve object memories without visibility information.

        Args:
            task_objects: Objects mentioned in task
            current_step: Current step number for recency scoring

        Returns:
            List of object memory documents
        """
        memories = []

        # Search for all task objects
        for obj_type in task_objects:
            query = f"Where is {obj_type}? {obj_type} location"
            results = self._store.query(
                query_text=query,
                n_results=3,
                memory_type=MemoryType.OBJECT
            )
            for result in results:
                result["priority"] = "medium"
                result["reason"] = f"Task object: {obj_type}"
            memories.extend(results)

        # Score by recency and deduplicate
        memories = self._score_and_deduplicate(memories, current_step)

        return memories[:self.MAX_OBJECT_MEMORIES]

    def _retrieve_observations_near_position(self,
                                              agent_position: Dict[str, float],
                                              current_step: int) -> List[Dict[str, Any]]:
        """Retrieve observations near the agent's current dead reckoning position.

        Args:
            agent_position: Dead reckoning estimated position
            current_step: Current step number

        Returns:
            List of observation memories
        """
        # Get recent observations
        results = self._store.query_by_type(
            memory_type=MemoryType.OBSERVATION,
            n_results=self.MAX_OBSERVATION_MEMORIES * 2,
            sort_by_recency=True
        )

        # Filter by proximity to current position (if position metadata exists)
        nearby = []
        for obs in results:
            meta = obs.get("metadata", {})
            obs_x = meta.get("agent_x")
            obs_z = meta.get("agent_z")

            if obs_x is not None and obs_z is not None:
                # Calculate distance from current position
                dx = agent_position["x"] - obs_x
                dz = agent_position["z"] - obs_z
                distance = (dx * dx + dz * dz) ** 0.5

                # Keep observations within 3 meters
                if distance < 3.0:
                    obs["distance_from_agent"] = distance
                    nearby.append(obs)
            else:
                # No position data, include anyway
                nearby.append(obs)

        # Sort by distance (nearest first) then by recency
        nearby.sort(key=lambda x: (x.get("distance_from_agent", 999), -x.get("metadata", {}).get("step", 0)))

        return nearby[:self.MAX_OBSERVATION_MEMORIES]

    def _extract_task_objects(self, task_description: str) -> Set[str]:
        """Extract object type names from task description.

        Args:
            task_description: The task goal text

        Returns:
            Set of potential object type names
        """
        # Common AI2Thor object types
        common_objects = {
            "Apple", "Bread", "Butter", "ButterKnife", "Cabinet",
            "CoffeeMachine", "CounterTop", "Cup", "DishSponge", "Drawer",
            "Egg", "Faucet", "Fork", "Fridge", "GarbageCan", "Kettle",
            "Knife", "Ladle", "Lettuce", "Microwave", "Mug", "Pan",
            "Plate", "Pot", "Potato", "Saltshaker", "Sink", "SinkBasin",
            "Spatula", "Spoon", "StoveBurner", "StoveKnob", "Toaster",
            "Tomato", "Bowl", "Vase", "Window", "Chair", "Table",
            "RemoteControl", "Television", "Sofa", "Pillow", "Book",
            "Laptop", "CellPhone", "AlarmClock", "Newspaper", "Box"
        }

        found_objects = set()

        # Look for exact matches (case-insensitive)
        task_lower = task_description.lower()
        for obj in common_objects:
            if obj.lower() in task_lower:
                found_objects.add(obj)

        # Also extract capitalized words that might be object names
        words = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', task_description)
        for word in words:
            if word not in {"Pick", "Put", "Move", "Go", "Open", "Close",
                           "Toggle", "Turn", "Call", "Done", "The", "Your",
                           "Task", "Stand", "Front", "So", "That", "You",
                           "Are", "Facing", "Directly", "Located"}:
                found_objects.add(word)

        return found_objects

    def _retrieve_object_memories(self,
                                   task_objects: Set[str],
                                   missing_objects: Set[str],
                                   current_step: int) -> List[Dict[str, Any]]:
        """Retrieve memories about task-relevant objects.

        Args:
            task_objects: All objects mentioned in task
            missing_objects: Objects not currently visible
            current_step: Current step number for recency scoring

        Returns:
            List of object memory documents
        """
        memories = []

        # Priority 1: Search for missing objects
        for obj_type in missing_objects:
            query = f"Where is {obj_type}? {obj_type} location"
            results = self._store.query(
                query_text=query,
                n_results=3,
                memory_type=MemoryType.OBJECT
            )
            for result in results:
                result["priority"] = "high"
                result["reason"] = f"Missing object: {obj_type}"
            memories.extend(results)

        # Priority 2: Search for all task objects
        for obj_type in task_objects:
            if obj_type not in missing_objects:
                query = f"{obj_type} state location"
                results = self._store.query(
                    query_text=query,
                    n_results=2,
                    memory_type=MemoryType.OBJECT
                )
                for result in results:
                    result["priority"] = "medium"
                    result["reason"] = f"Task object: {obj_type}"
                memories.extend(results)

        # Score by recency and deduplicate
        memories = self._score_and_deduplicate(memories, current_step)

        return memories[:self.MAX_OBJECT_MEMORIES]

    def _retrieve_failed_actions(self, current_step: int) -> List[Dict[str, Any]]:
        """Retrieve recent failed action memories.

        Args:
            current_step: Current step number

        Returns:
            List of failed action memories
        """
        # Get recent failures
        results = self._store.query_by_type(
            memory_type=MemoryType.ACTION,
            n_results=10,
            sort_by_recency=True
        )

        # Filter to only failures
        failures = [
            r for r in results
            if r["metadata"].get("success") is False
        ]

        # Only keep recent failures (within last 10 steps)
        recent_failures = [
            f for f in failures
            if current_step - f["metadata"].get("step", 0) <= 10
        ]

        return recent_failures[:self.MAX_ACTION_MEMORIES]

    def _retrieve_relations(self,
                            task_objects: Set[str],
                            current_step: int) -> List[Dict[str, Any]]:
        """Retrieve relationship memories involving task objects.

        Args:
            task_objects: Objects mentioned in task
            current_step: Current step number

        Returns:
            List of relation memories
        """
        memories = []

        # Search for relations involving task objects
        for obj_type in task_objects:
            query = f"{obj_type} is on in"
            results = self._store.query(
                query_text=query,
                n_results=3,
                memory_type=MemoryType.RELATION
            )
            memories.extend(results)

        # Also check for agent holding something
        holding_results = self._store.query(
            query_text="Agent holding",
            n_results=2,
            memory_type=MemoryType.RELATION
        )
        memories.extend(holding_results)

        return self._score_and_deduplicate(memories, current_step)[:self.MAX_RELATION_MEMORIES]

    def _retrieve_observations(self, current_step: int) -> List[Dict[str, Any]]:
        """Retrieve navigation/observation memories.

        Args:
            current_step: Current step number

        Returns:
            List of observation memories
        """
        # Get recent keyframe observations
        results = self._store.query_by_type(
            memory_type=MemoryType.OBSERVATION,
            n_results=self.MAX_OBSERVATION_MEMORIES,
            sort_by_recency=True
        )

        return results

    def _score_and_deduplicate(self,
                                memories: List[Dict[str, Any]],
                                current_step: int) -> List[Dict[str, Any]]:
        """Score memories by recency and remove duplicates.

        Args:
            memories: List of memory documents
            current_step: Current step number for recency scoring

        Returns:
            Deduplicated and scored memory list
        """
        seen_contents = set()
        unique_memories = []

        for mem in memories:
            content = mem.get("content", "")
            if content not in seen_contents:
                seen_contents.add(content)

                # Calculate recency score
                mem_step = mem.get("metadata", {}).get("step", 0)
                step_diff = max(1, current_step - mem_step)

                # Recency score: higher for more recent
                recency_score = 1.0 / step_diff

                # Combine with similarity distance if available
                distance = mem.get("distance", 0.5)
                similarity_score = 1.0 - distance  # Convert distance to similarity

                # Weighted combination
                mem["combined_score"] = (
                    self.RECENCY_WEIGHT * recency_score +
                    (1 - self.RECENCY_WEIGHT) * similarity_score
                )

                unique_memories.append(mem)

        # Sort by combined score (descending)
        unique_memories.sort(key=lambda x: x.get("combined_score", 0), reverse=True)

        return unique_memories

    def format_as_text(self, context: Dict[str, Any]) -> str:
        """Format retrieved context as readable text for LLM injection.

        Args:
            context: The retrieved context dictionary

        Returns:
            Formatted text string
        """
        lines = []

        # Object memories
        if context["object_memories"]:
            lines.append("## Object Locations:")
            for mem in context["object_memories"]:
                step = mem.get("metadata", {}).get("step", "?")
                lines.append(f"- {mem['content']} (Step {step})")

        # Relation memories
        if context["relation_memories"]:
            lines.append("\n## Object Relationships:")
            for mem in context["relation_memories"]:
                step = mem.get("metadata", {}).get("step", "?")
                lines.append(f"- {mem['content']} (Step {step})")

        # Failed actions
        if context["action_memories"]:
            lines.append("\n## Recent Failed Actions:")
            for mem in context["action_memories"]:
                step = mem.get("metadata", {}).get("step", "?")
                lines.append(f"- {mem['content']} (Step {step})")

        # Navigation history
        if context["observation_memories"]:
            lines.append("\n## Recent Positions Visited:")
            for mem in context["observation_memories"][:3]:
                step = mem.get("metadata", {}).get("step", "?")
                lines.append(f"- {mem['content']} (Step {step})")

        # Missing objects hint
        if context["missing_objects"]:
            lines.append(f"\n## Objects Needed But Not Visible:")
            lines.append(f"- {', '.join(context['missing_objects'])}")

        return "\n".join(lines) if lines else ""

    def generate_proactive_queries(self,
                                    task_description: str,
                                    env_state: EnvironmentState) -> List[str]:
        """Generate proactive queries for finding task-relevant information.

        This method identifies what information might be needed based on
        the task and current view, and generates appropriate queries.

        Args:
            task_description: The task goal
            env_state: Current environment state

        Returns:
            List of query strings
        """
        queries = []
        task_objects = self._extract_task_objects(task_description)
        visible_types = {obj.object_type for obj in env_state.visible_objects}

        # Query for each missing object
        for obj_type in task_objects - visible_types:
            queries.append(f"Where was {obj_type} last seen?")

        # If task involves putting something somewhere
        if "put" in task_description.lower():
            # Find receptacles mentioned
            receptacles = {"Fridge", "Pot", "Pan", "Bowl", "Plate",
                          "Sink", "StoveBurner", "CounterTop", "Cabinet",
                          "Microwave", "Drawer"}
            task_receptacles = task_objects & receptacles
            for rec in task_receptacles - visible_types:
                queries.append(f"Where is the {rec}?")

        return queries
