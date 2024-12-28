from typing import List, Union

import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage, SystemMessage,
    HumanMessage, ToolMessage,
)

from src.simulator import EnvironmentState, QueryReturn

class LLMAgent:
    def __init__(self, model: str, system_task: str,
                 human_task: str, tools: List[callable]):
        self._llm = ChatOpenAI(model=model, max_retries=2, temperature=0)
        self._system_task = system_task
        self._human_task = human_task
        self._llm = self._llm.bind_tools(tools, strict=True)
        self._first_message = True
        self._message_history = []

    def send_environment_feedback(
            self, env_feedback: Union[EnvironmentState, QueryReturn],
            return_message_id: str) -> AIMessage:
        if self._first_message:
            self._first_message = False
            encoded_img = env_feedback.agent_camera_view
            self._message_history.extend([
                SystemMessage(content=self._system_task),
                HumanMessage(content=self._human_task),
                HumanMessage(content=[
                    {"type": "text",
                     "text": "Here is a screenshot from the simulator"},
                    {"type": "image_url",
                     "image_url": {"url": encoded_img}}]),
            ])
            ai_response_msg = self._llm.invoke(self._message_history)
            self._message_history.append(ai_response_msg)
            return ai_response_msg
        else:
            encoded_img = env_feedback.agent_camera_view
            env_feedback.agent_camera_view = ''
            self._message_history.extend(
                [ToolMessage(
                    tool_call_id=return_message_id,
                    content=[
                        {"type": "text", "text": str(env_feedback)},
                    ],
                ),
                 HumanMessage(content=[
                    {"type": "text",
                     "text": "Here is a screenshot from the simulator"},
                    {"type": "image_url",
                     "image_url": {"url": encoded_img}}])
                ])
            ai_response_msg = self._llm.invoke(self._message_history)
            self._message_history.append(ai_response_msg)
            return ai_response_msg
