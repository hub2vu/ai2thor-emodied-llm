"""Represents a simulator backend for simulating the env and
the robot motion inside it.
"""
from typing import Dict, List, Callable, Any
from dataclasses import field

from ai2thor.controller import Controller
from ai2thor.server import Event
from langchain_core.tools import tool
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
        self._controller = Controller(scene=scene)

    def get_available_actions(self) -> List[Callable]:
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

    @staticmethod
    def extract_environment_state_from_event(event: Event) -> EnvironmentState:
        agent_position = event.metadata['agent']['position']
        agent_rotation = event.metadata['agent']['rotation']
        last_action_sucess = event.metadata['lastActionSuccess']
        error_message = event.metadata['errorMessage']
        env_state = EnvironmentState(agent_position, agent_rotation,
                                     last_action_sucess, error_message)
        return env_state

    @staticmethod
    def get_object_from_event(event: Event, object_id: str) -> Dict[str, Any]:
        for object_dict in event.metadata['objects']:
            if object_dict['objectId'] == object_id:
                return object_dict
        raise AttributeError(f'Object with id {object_id} doesn\'t exists in the current scene')

    @tool
    def move_back(self) -> EnvironmentState:
        event = self._controller.step(action="MoveBack")
        return self.extract_environment_state_from_event(event)

    @tool
    def move_ahead(self) -> EnvironmentState:
        event = self._controller.step(action="MoveAhead")
        return self.extract_environment_state_from_event(event)

    @tool
    def move_left(self) -> EnvironmentState:
        event = self._controller.step(action="MoveLeft")
        return self.extract_environment_state_from_event(event)

    @tool
    def move_right(self) -> EnvironmentState:
        event = self._controller.step(action="MoveRight")
        return self.extract_environment_state_from_event(event)

    @tool
    def rotate_left(self) -> EnvironmentState:
        event = self._controller.step(action="RotateLeft")
        return self.extract_environment_state_from_event(event)

    @tool
    def rotate_right(self) -> EnvironmentState:
        event = self._controller.step(action="RotateRight")
        return self.extract_environment_state_from_event(event)

    @tool
    def done(self) -> EnvironmentState:
        event = self._controller.step(action="Done")
        return self.extract_environment_state_from_event(event)

    @tool
    def pick_object(self, object_id: str) -> EnvironmentState:
        event = self._controller.step(
            action="PickupObject",
            objectId=object_id,
            forceAction=False,
            manualInteract=False)
        return self.extract_environment_state_from_event(event)

    @tool
    def put_object(self, object_id: str) -> EnvironmentState:
        event = self._controller.step(
            action="PutObject",
            objectId=object_id,
            forceAction=False,
            placeStationary=True)
        return self.extract_environment_state_from_event(event)

    @tool
    def open_object(self, object_id: str) -> EnvironmentState:
        event = self._controller.strp(
            action="OpenObject",
            objectId=object_id,
            openness=1,
            forceAction=False)
        return self.extract_environment_state_from_event(event)

    @tool
    def close_object(self, object_id: str) -> EnvironmentState:
        event = self._controller.strp(
            action="CloseObject",
            objectId=object_id,
            openness=1,
            forceAction=False)
        return self.extract_environment_state_from_event(event)

    @tool
    def toggle_object_on(self, object_id: str) -> EnvironmentState:
        event = self._controller.strp(
            action="ToggleObjectOn",
            objectId=object_id,
            forceAction=False)
        return self.extract_environment_state_from_event(event)

    @tool
    def toggle_object_off(self, object_id: str) -> EnvironmentState:
        event = self._controller.step(
            action="ToggleObjectOff",
            objectId=object_id,
            forceAction=False)
        return self.extract_environment_state_from_event(event)

    @tool
    def query_object(self, x: float, y: float) -> QueryReturn:
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
