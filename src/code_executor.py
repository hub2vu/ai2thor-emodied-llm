"""Code executor module for parsing and executing LLM-generated Python code."""

import re
from typing import Any, Optional, List
from io import StringIO
import sys


class CodeExecutionResult:
    """Represents the result of code execution."""

    def __init__(self, success: bool, result: Any = None,
                 error: Optional[str] = None, stdout: str = "",
                 visible_object_ids: Optional[List[str]] = None):
        """Initialize execution result.

        Args:
            success (bool): Whether execution was successful
            result (Any): The return value from executed code
            error (Optional[str]): Error message if execution failed
            stdout (str): Captured standard output during execution
            visible_object_ids (list): List of visible object IDs from the environment state
        """
        self.success = success
        self.result = result
        self.error = error
        self.stdout = stdout
        self.visible_object_ids = visible_object_ids if visible_object_ids is not None else []

    def __str__(self) -> str:
        if self.success:
            return f"Success: {self.result}\nOutput: {self.stdout}"
        else:
            return f"Error: {self.error}"


class CodeExecutor:
    """Executes LLM-generated Python code in a controlled environment."""

    def __init__(self, simulator):
        """Initialize the code executor.

        Args:
            simulator: The SimulatorBackend instance to expose to executed code
        """
        self.simulator = simulator

    def extract_code(self, llm_response: str) -> Optional[str]:
        """Extract Python code from LLM response.

        Looks for code in ```python blocks or direct function calls.

        Args:
            llm_response (str): The raw text response from the LLM

        Returns:
            Optional[str]: Extracted code, or None if no code found
        """
        # Try to find code in markdown code blocks
        code_block_pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Try to find direct function calls (simple pattern)
        # Look for lines that start with simulator.
        lines = llm_response.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('simulator.'):
                return stripped

        return None

    def execute_code(self, code: str) -> CodeExecutionResult:
        """Execute Python code with the simulator instance available.

        Args:
            code (str): The Python code to execute

        Returns:
            CodeExecutionResult: The result of execution
        """
        # Create restricted namespace with only simulator
        namespace = {
            'simulator': self.simulator,
            '__builtins__': {
                # Allow only safe built-ins
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'dict': dict,
                'list': list,
                'tuple': tuple,
                'range': range,
                'print': print,
            }
        }

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            # Execute the code and capture the result
            # Use eval for single expressions, exec for statements
            result = None
            code = code.strip()

            # Try to evaluate as expression first (for single line calls)
            try:
                result = eval(code, namespace)
            except SyntaxError:
                # If eval fails, use exec (for multi-line or statements)
                exec(code, namespace)
                # Get the result (if any variable named 'result' was set)
                result = namespace.get('result', None)

            # Restore stdout and get captured output
            sys.stdout = old_stdout
            stdout_content = captured_output.getvalue()

            # Parse the result to extract visible object IDs and format message
            visible_object_ids = []
            formatted_result = result

            if result is not None and hasattr(result, 'visible_objects'):
                visible_object_ids = [obj.object_id for obj in result.visible_objects]

                # Create formatted message
                if visible_object_ids:
                    object_list = '\n'.join(f'- {obj_id}' for obj_id in visible_object_ids)
                    formatted_result = f"The current visible objects are (object_id=type|x_loc|y_loc|z_loc):\n{object_list}"
                else:
                    formatted_result = "The current visible objects are:\n(No visible objects)"

            return CodeExecutionResult(
                success=True,
                result=result,
                stdout=stdout_content,
                visible_object_ids=visible_object_ids
            )

        except Exception as e:
            # Restore stdout
            sys.stdout = old_stdout

            # Return error information
            error_msg = f"{type(e).__name__}: {str(e)}"
            return CodeExecutionResult(
                success=False,
                error=error_msg
            )

    def parse_and_execute(self, llm_response: str) -> CodeExecutionResult:
        """Extract code from LLM response and execute it.

        Args:
            llm_response (str): The raw text response from the LLM

        Returns:
            CodeExecutionResult: The result of execution
        """
        code = self.extract_code(llm_response)

        if code is None:
            return CodeExecutionResult(
                success=False,
                error="No executable code found in LLM response"
            )

        return self.execute_code(code)
