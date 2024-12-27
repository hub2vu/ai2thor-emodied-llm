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

    def send_environment_feedback(
            self, env_feedback: Union[EnvironmentState, QueryReturn],
            return_message_id: str) -> AIMessage:
        if self._first_message:
            self._first_message = False
            return self._llm.invoke([
                SystemMessage(content=self._system_task),
                HumanMessage(content=self._human_task),
                ToolMessage(
                    tool_call_id=return_message_id,
                    content=env_feedback
                )
            ])
        else:
            return self._llm.invoke([
                ToolMessage(
                    tool_call_id=return_message_id,
                    content=env_feedback
                )
            ])
