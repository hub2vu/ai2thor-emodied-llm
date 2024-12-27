"""Represents a simulator backend for simulating the env and
the robot motion inside it.
"""
from typing import Dict, List, Callable, Any, Tuple
from dataclasses import field
from io import BytesIO
import base64

import numpy as np
from PIL import Image
from ai2thor.controller import Controller
from ai2thor.server import Event
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from pydantic.dataclasses import dataclass


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
    agent_camera_view: str = field(metadata={
        "description": "The agent camera view data url"})


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


class SimulatorBackend:
    def __init__(self, scene: str):
        """Represents the simulator backend so that
        the agent can interact with.

        Args:
            scene (str): The scene name to use.
        """
        self._controller = Controller(scene=scene)

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
            self.query_object,
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
            self, ai_message: AIMessage) -> Tuple[EnvironmentState, str]:
        """Executes the action from the specified message.

        Args:
            ai_message (AIMessage): The AI message to execute the calls from.

        Returns:
            Tuple[EnvironmentState, str]: The environment feedback
                and the executed action id.
        """
        tool_callback_dict = ai_message.tool_calls[0]
        call_id = tool_callback_dict["id"]
        func_name = tool_callback_dict['name']
        func_args = tool_callback_dict['args']
        func: Callable = getattr(self, func_name)
        env_feedback = func(**func_args)
        return (env_feedback, call_id)

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
        env_state = EnvironmentState(agent_position, agent_rotation,
                                     last_action_sucess, error_message,
                                     png_data_url)
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
    def encode_img_as_base64_png_data_url(img: np.ndarray) -> str:
        """Encode a numpy array as png data url encoded in base64 string.

        Args:
            img (np.ndarray): The image to encode.

        Returns:
            str: The encoded image data url.
        """
        pil_img = Image.fromarray(img)
        buffered = BytesIO()
        pil_img.save(buffered, format='PNG')
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        data_url = f"data:image/png;base64,{img_base64}"
        return data_url

    def initailize_simulator(self) -> EnvironmentState:
        event = self._controller.step(action="Initialize")
        return self.extract_environment_state_from_event(event)

    @tool
    def move_back(self) -> EnvironmentState:
        """Moves the agent backward by 0.25 meters

        Returns:
            EnvironmentState: The environment state after executing the action.
        """
        event = self._controller.step(action="MoveBack")
        return self.extract_environment_state_from_event(event)

    @tool
    def move_ahead(self) -> EnvironmentState:
        """Moves the agent forward by 0.25 meters

        Returns:
            EnvironmentState: The environment state after executing the action.
        """
        event = self._controller.step(action="MoveAhead")
        return self.extract_environment_state_from_event(event)

    @tool
    def move_left(self) -> EnvironmentState:
        """Moves the agent left by 0.25 meters

        Returns:
            EnvironmentState: The environment state after executing the action.
        """
        event = self._controller.step(action="MoveLeft")
        return self.extract_environment_state_from_event(event)

    @tool
    def move_right(self) -> EnvironmentState:
        """Moves the agent right by 0.25 meters

        Returns:
            EnvironmentState: The environment state after executing the action.
        """
        event = self._controller.step(action="MoveRight")
        return self.extract_environment_state_from_event(event)

    @tool
    def rotate_left(self) -> EnvironmentState:
        """Rotates the agent left by 90 degrees

        Returns:
            EnvironmentState: The environment state after executing the action.
        """
        event = self._controller.step(action="RotateLeft")
        return self.extract_environment_state_from_event(event)

    @tool
    def rotate_right(self) -> EnvironmentState:
        """Rotates the agent right by 90 degrees

        Returns:
            EnvironmentState: The environment state after executing the action.
        """
        event = self._controller.step(action="RotateRight")
        return self.extract_environment_state_from_event(event)

    @tool
    def done(self) -> EnvironmentState:
        """Indicates that the task has been completed.

        Returns:
            EnvironmentState: The environment state after executing the action.
        """
        event = self._controller.step(action="Done")
        return self.extract_environment_state_from_event(event)

    @tool
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
        return self.extract_environment_state_from_event(event)

    @tool
    def put_object(self, object_id: str) -> EnvironmentState:
        """Puts an object that has been picked in the agents hand.

        Args:
            object_id (str): The object id of the object
                that is on the agent hand.

        Returns:
            EnvironmentState: The environment state after executing the action.
        """
        event = self._controller.step(
            action="PutObject",
            objectId=object_id,
            forceAction=False,
            placeStationary=True)
        return self.extract_environment_state_from_event(event)

    @tool
    def open_object(self, object_id: str) -> EnvironmentState:
        """Opens an object like a frdige or microwave.

        Args:
            object_id (str): The closed The object id
                for the object that needs to be opened.

        Returns:
            EnvironmentState: The environment state after executing the action.
        """
        event = self._controller.strp(
            action="OpenObject",
            objectId=object_id,
            openness=1,
            forceAction=False)
        return self.extract_environment_state_from_event(event)

    @tool
    def close_object(self, object_id: str) -> EnvironmentState:
        """Closes an object like a frdige or microwave.

        Args:
            object_id (str): The object id for the object
                that needs to be closed.

        Returns:
            EnvironmentState: The environment state after executing the action.
        """
        event = self._controller.strp(
            action="CloseObject",
            objectId=object_id,
            openness=1,
            forceAction=False)
        return self.extract_environment_state_from_event(event)

    @tool
    def toggle_object_on(self, object_id: str) -> EnvironmentState:
        """Toggles on an object like a microwave.

        Args:
            object_id (str): The object id for the object to toggle on.

        Returns:
            EnvironmentState: The environment state after executing the action.
        """
        event = self._controller.strp(
            action="ToggleObjectOn",
            objectId=object_id,
            forceAction=False)
        return self.extract_environment_state_from_event(event)

    @tool
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
        return self.extract_environment_state_from_event(event)

    @tool
    def query_object(self, x: float, y: float) -> QueryReturn:
        """Queries the object specified by the x, y coordinates
        in the agent view.

        Args:
            x (float): The normalized x-coodinate of the object
                relative to the top-left corner of the agent view image.
                This is a normalized coordinates, so it has a range [0, 1].
            y (float): The normalized y-coodinate of the object
                relative to the top-left corner of the agent view image.
                This is a normalized coordinates, so it has a range [0, 1].

        Returns:
            QueryReturn: The output of quering the object.
        """
        event = self._controller.step(
            action="GetObjectInFrame",
            x=x,
            y=y,
            checkVisible=True
        )
        object_id = event.metadata["actionReturn"]
        if object_id is None or object_id == '':
            return QueryReturn('', {}, False, False)
        else:
            object_dict = self.get_object_from_event(object_id)
            is_visible = object_dict['visible']
            object_position = object_dict['position']
            query_return = QueryReturn(
                object_id, object_position, is_visible, True)
            return query_return
