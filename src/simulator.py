"""Represents a simulator backend for simulating the env and
the robot motion inside it.
"""

from ai2thor.controller import Controller


class SimulatorBackend:
    def __init__(self, scene: str):
        self._controller = Controller(scene=scene)

    def move_back(self):
        # Returns:
        # agent_postion + rotation
        # last_action_success
        # error_message
        event = self._controller.step(action="MoveBack")

    def move_ahead(self):
        event = self._controller.step(action="MoveAhead")

    def move_left(self):
        event = self._controller.step(action="MoveLeft")

    def move_right(self):
        event = self._controller.step(action="MoveRight")

    def rotate_left(self):
        event = self._controller.step(action="RotateLeft")

    def rotate_right(self):
        event = self._controller.step(action="RotateRight")

    def done(self):
        event = self._controller.step(action="Done")

    def pick_object(self, object_id: str):
        event = self._controller.step(
            action="PickupObject",
            objectId=object_id,
            forceAction=False,
            manualInteract=False)

    def put_object(self, object_id: str):
        event = self._controller.step(
            action="PutObject",
            objectId=object_id,
            forceAction=False,
            placeStationary=True)

    def open_object(self, object_id: str):
        event = self._controller.strp(
            action="OpenObject",
            objectId=object_id,
            openness=1,
            forceAction=False)

    def close_object(self, object_id: str):
        event = self._controller.strp(
            action="CloseObject",
            objectId=object_id,
            openness=1,
            forceAction=False)

    def toggle_object_on(self, object_id: str):
        event = self._controller.strp(
            action="ToggleObjectOn",
            objectId=object_id,
            forceAction=False)

    def toggle_object_off(self, object_id: str):
        event = self._controller.step(
            action="ToggleObjectOff",
            objectId=object_id,
            forceAction=False)

    def query_object(self, x: float, y: float):
        # Return 
        # - object_id
        # - object_position
        # - is_visible
        # - is_valid
        query = self._controller.step(
            action="GetObjectInFrame",
            x=x,
            y=y,
            checkVisible=True
        )
        object_id = query.metadata["actionReturn"]