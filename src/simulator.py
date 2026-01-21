"""Represents a simulator backend for simulating the env and
the robot motion inside it.
"""
from typing import Dict, List, Callable, Any, Tuple
from dataclasses import field
from io import BytesIO
import base64
import math

import numpy as np
from PIL import Image
from ai2thor.controller import Controller
from ai2thor.server import Event
from pydantic.dataclasses import dataclass
from pydantic import BaseModel, Field


@dataclass
class SimulatorObject:
    """A Class that describes the state of an object within the simulator.
    """
    position: Dict[str, float] = field(metadata={
        "description": "Contains the x, y, z "
        "position of the object relative to the world coordinates in meters"})
    object_id: str = field(metadata={
        "description": "The object id that can be used to interact "
        "with this object"})
    object_type: str = field(metadata={
        "description": "The class type of the object"})
    # Enhanced fields for RAG memory
    parent_receptacles: List[str] = field(default_factory=list, metadata={
        "description": "List of receptacle object IDs that contain this object"})
    is_open: bool = field(default=False, metadata={
        "description": "Whether the object is open (for openable objects)"})
    is_toggled: bool = field(default=False, metadata={
        "description": "Whether the object is toggled on (for toggleable objects)"})
    is_pickupable: bool = field(default=False, metadata={
        "description": "Whether the object can be picked up"})
    is_receptacle: bool = field(default=False, metadata={
        "description": "Whether the object can contain other objects"})


@dataclass
class EnvironmentState:
    """A Class that returns the environment state after executing an
    action by the agent.
    """
    agent_position: Dict[str, float] = field(metadata={
        "description": "Contains the x, y, z "
        "position of the agent relative to the world coordinates in meters"})
    agent_rotation: Dict[str, float] = field(metadata={
        "description": "Contains the rotations around x, y, z of the agent "
        "realtive to the world coordinates in degrees"})
    last_action_success: bool = field(metadata={
        "description": "if True then the last action executed by "
        "the agent was successfull, otherwise False"})
    error_message: str = field(metadata={
        "description": "if the last action was not successfull then this is "
        "the resulted error message due to action execution"})
    visible_objects: List[SimulatorObject] = field(metadata={
        "description": "A list of the visible objects to the robot "
        "agent inside the simulator"})
    agent_camera_view: str = field(metadata={
        "description": "The agent camera view data url"})

    def __str__(self) -> str:
        return (f"EvironmentState(agent_position={self. agent_position}, "
                f"agent_rotation={self.agent_rotation}, "
                f"last_action_success={self.last_action_success}, "
                f"error_message='{self.error_message}', "
                f"visible_objects={self.visible_objects})")


@dataclass
class QueryReturn:
    """Represents the Return value of quering an object using x, y coordinates.
    """
    object_id: str = field(metadata={
        "description": "Contains the object_id from the simulator"})
    object_position: Dict[str, float] = field(metadata={
        "description": "Contains the x, y, z "
        "position of the object relative to the world coordinates in meters"})
    is_visible: bool = field(metadata={
        "description": "True if the object is visible and within the range "
        "of interaction of the agent, otherwise False"})
    is_valid: bool = field(metadata={
        "description": "True if there is a valid object "
        "returned by the query"})
    agent_camera_view: str = field(metadata={
        "description": "The agent camera view data url"})


class NoArgsSchema(BaseModel):
    pass


class PickObject(BaseModel):
    """
    Pick up an object and place it in the agents hand, the
    object must be visible and within the agents reach.
    """
    object_id: str = Field(
        metadata={"description": "The object id of the object to pick"}
    )


class PutObject(BaseModel):
    """
    Puts an object that has been picked in the agents hand.
    """
    target_id: str = Field(
        metadata={"description": "The object id of the "
                  "target receptable object "
                  "(object which will recieve the object "
                  "from the agent hands)"}
    )


class OpenObject(BaseModel):
    """
    Opens an object like a fridge or microwave.
    """
    object_id: str = Field(
        metadata={"description": "The object id for the "
                  "object that needs to be opened."}
    )


class CloseObject(BaseModel):
    """
    Closes an object like a fridge or microwave.
    """
    object_id: str = Field(
        metadata={"description": "The object id for "
                  "the object that needs to be closed."}
    )


class ToggleObjectOn(BaseModel):
    """
    Toggles on an object like a microwave.
    """
    object_id: str = Field(
        metadata={"description": "The object id for the object to toggle on."}
    )


class ToggleObjectOff(BaseModel):
    """
    Toggles off an object like a microwave.
    """
    object_id: str = Field(
        metadata={"description": "The object id for the object to toggle off."}
    )


class QueryObject(BaseModel):
    """
    Queries the object specified by the x, y coordinates
    in the agent view.
    """
    x: float = Field(
        metadata={"description": "The normalized x-coordinate of the object "
                  "relative to the top-left corner of the agent view image. "
                  "This is a normalized coordinate, so it has a range [0, 1]."}
    )
    y: float = Field(
        metadata={"description": "The normalized y-coordinate of the object "
                  "relative to the top-left corner of the agent view image. "
                  "This is a normalized coordinate, so it has a range [0, 1]."}
    )


class SimulatorBackend:
    # Objects that require direct facing for interaction (door opens towards agent)
    ALIGNMENT_REQUIRED_OBJECTS = {"Fridge", "Cabinet", "Microwave"}
    ALIGNMENT_THRESHOLD_DEGREES = 30.0

    def __init__(self, scene: str):
        """Represents the simulator backend so that
        the agent can interact with.

        Args:
            scene (str): The scene name to use.
        """
        self._rotation_degrees = 90
        self._last_event = None  # Track last event for alignment checks
        self._controller = Controller(
            scene=scene,
            width=800,
            height=600,
            rotateStepDegrees=self._rotation_degrees,
            snapToGrid=True,
            renderImage=True,           # Explicitly enable image rendering
            visibilityDistance=1.5      # Match interaction distance
        )

    def get_available_actions(self) -> List[Callable]:
        """Returns the available actions within the simulator
        that the agent can take.

        Returns:
            List[Callable]: The list of available actions to take.
        """
        return [
            self.move_back,
            self.move_ahead,
            self.move_left,
            self.move_right,
            self.rotate_left,
            self.rotate_right,
            self.close_object,
            self.open_object,
            self.put_object,
            self.pick_object,
            # self.query_object,
            self.done,
            self.toggle_object_on,
            self.toggle_object_off,
        ]

    def get_key_event_vs_action(self) -> Dict[str, Callable]:
        """Returns the event vs action dictionary

        Returns:
            Dict[str, Callable]: The event name vs action dictionary.
        """
        return {
            "s": self.move_back,
            "w": self.move_ahead,
            "a": self.move_left,
            "d": self.move_right,
            "left": self.rotate_left,
            "right": self.rotate_right,
            "c": self.close_object,
            "o": self.open_object,
            "p": self.put_object,
            "i": self.pick_object,
            "q": self.query_object,
            "n": self.done,
            "t": self.toggle_object_on,
            "j": self.toggle_object_off,
        }

    def execute_action(
            self, env_state: EnvironmentState) -> EnvironmentState:
        """Returns the environment state (used for compatibility).

        Args:
            env_state (EnvironmentState): The environment state from action execution.

        Returns:
            EnvironmentState: The environment feedback.
        """
        return env_state

    @staticmethod
    def extract_environment_state_from_event(event: Event) -> EnvironmentState:
        """Extracts the environment state from an event and places it
        to the form of EnvironmentState instance.

        Args:
            event (Event): The Event due to an agent action.

        Returns:
            EnvironmentState: The instance that abstracts the
                environment state.
        """
        agent_position = event.metadata['agent']['position']
        agent_rotation = event.metadata['agent']['rotation']
        last_action_sucess = event.metadata['lastActionSuccess']
        error_message = event.metadata['errorMessage']
        img_frame = event.frame
        png_data_url = SimulatorBackend.encode_img_as_base64_png_data_url(
            img_frame)
        visible_objects = SimulatorBackend.get_visible_objects_from_event(
            event)
        env_state = EnvironmentState(agent_position, agent_rotation,
                                     last_action_sucess, error_message,
                                     visible_objects, png_data_url)
        return env_state

    @staticmethod
    def get_object_from_event(event: Event, object_id: str) -> Dict[str, Any]:
        """Returns an object specified by an object_id from the event which
        contained the environment attributes.

        Args:
            event (Event): The event resulted from the agent taking an action.
            object_id (str): The object id to search for.

        Raises:
            AttributeError: If no object is found with the specified id.

        Returns:
            Dict[str, Any]: The found object dict.
        """
        for object_dict in event.metadata['objects']:
            if object_dict['objectId'] == object_id:
                return object_dict
        raise AttributeError(f'Object with id {object_id} doesn\'t '
                             'exists in the current scene')

    @staticmethod
    def get_visible_objects_from_event(event: Event) -> List[SimulatorObject]:
        """Extract visible objects with enhanced metadata from AI2-THOR event.

        Args:
            event (Event): The AI2-THOR event containing object metadata.

        Returns:
            List[SimulatorObject]: List of visible objects with full state info.
        """
        visible_objects = []
        for object_dict in event.metadata['objects']:
            if object_dict['visible']:
                # Extract parent receptacles (objects that contain this object)
                parent_receptacles = object_dict.get('parentReceptacles') or []

                obj = SimulatorObject(
                    position=object_dict['position'],
                    object_id=object_dict['objectId'],
                    object_type=object_dict['objectType'],
                    parent_receptacles=parent_receptacles,
                    is_open=object_dict.get('isOpen', False),
                    is_toggled=object_dict.get('isToggled', False),
                    is_pickupable=object_dict.get('pickupable', False),
                    is_receptacle=object_dict.get('receptacle', False)
                )
                visible_objects.append(obj)
        return visible_objects

    @staticmethod
    def encode_img_as_base64_png_data_url(img: np.ndarray) -> str:
        """Encode a numpy array as png data url encoded in base64 string.

        Args:
            img (np.ndarray): The image to encode.

        Returns:
            str: The encoded image data url.
        """
        pil_img = Image.fromarray(img)
        buffered = BytesIO()
        pil_img.save(buffered, format='JPEG')
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{img_base64}"
        return data_url

    def _is_facing_object_directly(self, object_id: str,
                                     threshold_degrees: float = 30.0) -> Tuple[bool, str]:
        """Check if the agent is facing the object directly (within threshold angle).

        This is important for objects like Fridge where the door opens towards
        the agent - opening from a diagonal angle causes occlusion issues.

        Args:
            object_id: The object ID to check alignment with
            threshold_degrees: Maximum allowed angle deviation from direct facing

        Returns:
            Tuple of (is_aligned, error_message)
        """
        if not self._last_event:
            return True, ""  # No event info available, skip check

        # 1. Find target object position
        target_obj = None
        for obj in self._last_event.metadata['objects']:
            if obj['objectId'] == object_id:
                target_obj = obj
                break

        if not target_obj:
            return False, "Object not found in scene metadata."

        # 2. Get agent position and rotation
        agent_pos = self._last_event.metadata['agent']['position']
        agent_rot_y = self._last_event.metadata['agent']['rotation']['y']

        # 3. Calculate vector from agent to object
        dx = target_obj['position']['x'] - agent_pos['x']
        dz = target_obj['position']['z'] - agent_pos['z']

        # 4. Calculate agent's look direction vector
        # AI2-THOR: 0°=North(+z), 90°=East(+x), 180°=South(-z), 270°=West(-x)
        rad = math.radians(agent_rot_y)
        look_dx = math.sin(rad)
        look_dz = math.cos(rad)

        # 5. Calculate angle between the two vectors using dot product
        dot_product = (dx * look_dx) + (dz * look_dz)
        mag_dist = math.sqrt(dx * dx + dz * dz)
        mag_look = 1.0  # Unit vector

        if mag_dist == 0:
            return True, ""  # Object at same position as agent

        cos_angle = dot_product / (mag_dist * mag_look)
        # Clamp to handle floating point errors
        cos_angle = max(min(cos_angle, 1.0), -1.0)

        angle_degrees = math.degrees(math.acos(cos_angle))

        if angle_degrees > threshold_degrees:
            return False, (
                f"You are viewing the object from a diagonal angle ({angle_degrees:.1f}°). "
                f"Please move to face it directly (within {threshold_degrees}°)."
            )

        return True, ""

    def _force_render_update(self) -> None:
        """Force Unity to flush the render buffer by executing a Pass action.

        This is needed on Linux where Unity's VSync can cause the display to
        only update every 2 actions due to double-buffering.
        """
        self._controller.step(action="Pass")

    def initailize_simulator(self) -> EnvironmentState:
        event = self._controller.step(action="Initialize")
        self._last_event = event  # Store for alignment checks
        self._force_render_update()  # Force immediate display update
        return self.extract_environment_state_from_event(event)

    def move_back(self) -> EnvironmentState:
        """Moves the agent backward by 0.25 meters

        Returns:
            EnvironmentState: The environment state after executing the action.
        """
        event = self._controller.step(action="MoveBack")
        self._last_event = event
        self._force_render_update()  # Force immediate display update
        return self.extract_environment_state_from_event(event)

    def move_ahead(self) -> EnvironmentState:
        """Moves the agent forward by 0.25 meters

        Returns:
            EnvironmentState: The environment state after executing the action.
        """
        event = self._controller.step(action="MoveAhead")
        self._last_event = event
        self._force_render_update()  # Force immediate display update
        return self.extract_environment_state_from_event(event)

    def move_left(self) -> EnvironmentState:
        """Moves the agent left by 0.25 meters

        Returns:
            EnvironmentState: The environment state after executing the action.
        """
        event = self._controller.step(action="MoveLeft")
        self._last_event = event
        self._force_render_update()  # Force immediate display update
        return self.extract_environment_state_from_event(event)

    def move_right(self) -> EnvironmentState:
        """Moves the agent right by 0.25 meters

        Returns:
            EnvironmentState: The environment state after executing the action.
        """
        event = self._controller.step(action="MoveRight")
        self._last_event = event
        self._force_render_update()  # Force immediate display update
        return self.extract_environment_state_from_event(event)

    def rotate_left(self) -> EnvironmentState:
        """Rotates the agent left by 90 degrees

        Returns:
            EnvironmentState: The environment state after executing the action.
        """
        event = self._controller.step(
            action="RotateLeft",
            degrees=self._rotation_degrees)
        self._last_event = event
        self._force_render_update()  # Force immediate display update
        return self.extract_environment_state_from_event(event)

    def rotate_right(self) -> EnvironmentState:
        """Rotates the agent right by 90 degrees

        Returns:
            EnvironmentState: The environment state after executing the action.
        """
        event = self._controller.step(
            action="RotateRight",
            degrees=self._rotation_degrees)
        self._last_event = event
        self._force_render_update()  # Force immediate display update
        return self.extract_environment_state_from_event(event)

    def done(self) -> EnvironmentState:
        """Indicates that the task has been completed.

        Returns:
            EnvironmentState: The environment state after executing the action.
        """
        global should_exit
        event = self._controller.step(action="Done")
        self._last_event = event
        self._force_render_update()  # Force immediate display update
        print("Done Action has been called .. mission finished .. exiting!!.")
        self._controller.stop()
        should_exit = True
        exit(0)
        return self.extract_environment_state_from_event(event)

    def pick_object(self, object_id: str) -> EnvironmentState:
        """Pick up an object and place it in the agents hand, the
        object must be visible and within the agents reach.

        Args:
            object_id (str): The object id of the object to pick

        Returns:
            EnvironmentState: The environment state after executing the action.
        """
        event = self._controller.step(
            action="PickupObject",
            objectId=object_id,
            forceAction=False,
            manualInteract=False)
        self._last_event = event
        self._force_render_update()  # Force immediate display update
        return self.extract_environment_state_from_event(event)

    def put_object(self, target_id: str) -> EnvironmentState:
        """Puts an object that has been picked in the agents hand.

        Args:
            target_id (str): The object id of the receptable object
                like fridge or cabinet or microwave.

        Returns:
            EnvironmentState: The environment state after executing the action.
        """
        event = self._controller.step(
            action="PutObject",
            objectId=target_id,
            forceAction=False,
            placeStationary=True)
        self._last_event = event
        self._force_render_update()  # Force immediate display update
        return self.extract_environment_state_from_event(event)

    def open_object(self, object_id: str) -> EnvironmentState:
        """Opens an object like a fridge or microwave.

        For objects with doors that open towards the agent (Fridge, Cabinet, Microwave),
        the agent must be facing the object directly to avoid occlusion issues.

        Args:
            object_id (str): The object id for the object that needs to be opened.

        Returns:
            EnvironmentState: The environment state after executing the action.
        """
        # Check if this object requires alignment check
        object_type = object_id.split("|")[0] if "|" in object_id else object_id
        if object_type in self.ALIGNMENT_REQUIRED_OBJECTS:
            is_aligned, error_msg = self._is_facing_object_directly(
                object_id, self.ALIGNMENT_THRESHOLD_DEGREES
            )
            if not is_aligned:
                # Not aligned: return error without executing action
                current_state = self.extract_environment_state_from_event(self._last_event)
                current_state.last_action_success = False
                current_state.error_message = f"Alignment Error: {error_msg}"
                return current_state

        # Execute the open action
        event = self._controller.step(
            action="OpenObject",
            objectId=object_id,
            openness=1,
            forceAction=False)
        self._last_event = event
        self._force_render_update()  # Force immediate display update
        return self.extract_environment_state_from_event(event)

    def close_object(self, object_id: str) -> EnvironmentState:
        """Closes an object like a fridge or microwave.

        Args:
            object_id (str): The object id for the object
                that needs to be closed.

        Returns:
            EnvironmentState: The environment state after executing the action.
        """
        event = self._controller.step(
            action="CloseObject",
            objectId=object_id,
            forceAction=False)
        self._last_event = event
        self._force_render_update()  # Force immediate display update
        return self.extract_environment_state_from_event(event)

    def toggle_object_on(self, object_id: str) -> EnvironmentState:
        """Toggles on an object like a microwave.

        Args:
            object_id (str): The object id for the object to toggle on.

        Returns:
            EnvironmentState: The environment state after executing the action.
        """
        event = self._controller.step(
            action="ToggleObjectOn",
            objectId=object_id,
            forceAction=False)
        self._last_event = event
        self._force_render_update()  # Force immediate display update
        return self.extract_environment_state_from_event(event)

    def toggle_object_off(self, object_id: str) -> EnvironmentState:
        """Toggles off an object like a microwave.

        Args:
            object_id (str): The object id for the object to toggle off.

        Returns:
            EnvironmentState: The environment state after executing the action.
        """
        event = self._controller.step(
            action="ToggleObjectOff",
            objectId=object_id,
            forceAction=False)
        self._last_event = event
        self._force_render_update()  # Force immediate display update
        return self.extract_environment_state_from_event(event)

    def slice_object(self, object_id: str) -> EnvironmentState:
        """Slices an object like bread, potato, or other sliceable items.

        Args:
            object_id (str): The object id for the object to slice.
                The object must be visible and within reach.

        Returns:
            EnvironmentState: The environment state after executing the action.
        """
        event = self._controller.step(
            action="SliceObject",
            objectId=object_id,
            forceAction=False)
        self._last_event = event
        self._force_render_update()  # Force immediate display update
        return self.extract_environment_state_from_event(event)

    # @tool(args_schema=QueryObject)
    # def query_object(self, x: float, y: float) -> QueryReturn:
    #     """Queries the object specified by the x, y coordinates
    #     in the agent view.

    #     Args:
    #         x (float): The normalized x-coodinate of the object
    #             relative to the top-left corner of the agent view image.
    #             This is a normalized coordinates, so it has a range [0, 1].
    #         y (float): The normalized y-coodinate of the object
    #             relative to the top-left corner of the agent view image.
    #             This is a normalized coordinates, so it has a range [0, 1].

    #     Returns:
    #         QueryReturn: The output of quering the object.
    #     """
    #     event = self._controller.step(
    #         action="GetObjectInFrame",
    #         x=x,
    #         y=y,
    #         checkVisible=True
    #     )
    #     object_id = event.metadata["actionReturn"]
    #     img_frame = event.frame
    #     png_data_url = SimulatorBackend.encode_img_as_base64_png_data_url(
    #         img_frame)
    #     if object_id is None or object_id == '':
    #         return QueryReturn('', {}, False, False, png_data_url)
    #     else:
    #         object_dict = self.get_object_from_event(object_id)
    #         is_visible = object_dict['visible']
    #         object_position = object_dict['position']
    #         query_return = QueryReturn(
    #             object_id, object_position, is_visible, True,
    #             png_data_url)
    #         return query_return
