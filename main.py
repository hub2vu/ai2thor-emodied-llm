"""Represents the main entry point of the project.
"""

from typing import Union
import argparse
import yaml
import time
from pynput import keyboard

from src.simulator import (
    SimulatorBackend, EnvironmentState,
    QueryReturn
)
from src.llm_agent import LLMAgent
from src.code_executor import CodeExecutor


def parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='Simulates an embodied AI using a simulator '
        'backend and an LLM for controlling it')
    parser.add_argument(
        '--config-file',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    return parser.parse_args()


def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its content as a Python dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: A dictionary containing the parsed YAML data.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError as e:
        print(f"Error: The file at {file_path} was not found.")
        raise e
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file at {file_path}.")
        raise e


def process_ai_step():
    """
    Processes one AI step: sends feedback to LLM and executes generated code.
    Returns False if should exit, True otherwise.
    """
    global env_feedback, execution_result
    global agent, simulator, code_executor, ai_message

    # Send environment feedback to LLM
    ai_message = agent.send_environment_feedback(
        env_feedback, execution_result)

    # Extract content as string
    content = ai_message.content
    if isinstance(content, list):
        # If content is a list, join text parts
        content = " ".join([part if isinstance(part, str) else str(part.get('text', ''))
                          for part in content])

    print(f"AI Message: {content}")
    print("\n")

    # Parse and execute the generated code
    exec_result = code_executor.parse_and_execute(content)
    time.sleep(1.0)

    if not exec_result.success:
        print(f"Execution Error: {exec_result.error}")
        execution_result = str(exec_result)
        # Check if it's a "no code found" error - might indicate task completion
        if exec_result.error and "No executable code found" in exec_result.error:
            print("No code found - task may be complete. Exiting...")
            return False
    else:
        # Execution succeeded - result should be an EnvironmentState
        if exec_result.result is not None:
            temp_env_feedback = exec_result.result
            print(f"Action executed successfully")
            # print(f"Env Feedback: {temp_env_feedback}")
            print("--------------------------------------------------")
            # Update the environment feedback
            env_feedback = temp_env_feedback
            execution_result = None  # Clear execution result on success
        else:
            print("Warning: Code executed but no result returned")
            execution_result = "Code executed but returned None"

    return True


def handle_key_press(
        key: keyboard.KeyCode):
    """
    Handles the key press events for 'esc' and 'c'.
    """
    global should_exit, last_key_time, key_pressed

    if key == keyboard.Key.esc:  # Exit if the 'Esc' key is pressed
        print("Exiting...")
        should_exit = True
        return False  # Stop the listener
    elif hasattr(key, 'char') and key.char == 'c':  # Check for 'c' key
        print("Continuing...")
        key_pressed = True
        last_key_time = time.time()

        if not process_ai_step():
            should_exit = True
            return False

    elif hasattr(key, 'char'):  # Handle other key presses
        print(f"Wrong key press: {key.char}, Skipping...")
        print("Press 'Esc' to exit")
        print("Press 'c' to continue (or wait 1 second for auto-continue)")


def main() -> None:
    """Main function to run the script."""
    # I had to do that to make them accessible from the listener function,
    # I might have used a class
    # and stored them in the class state instead
    global env_feedback, execution_result
    global agent, simulator, code_executor, ai_message
    global should_exit, last_key_time, key_pressed

    args = parse_args()

    # Access the config file path
    config_file = args.config_file
    config = read_yaml_file(config_file)
    simulator_config = config['simulator_config']
    simulator = SimulatorBackend(**simulator_config)
    agent_config = config['llm_config']

    # Create LLM agent (no longer needs tools parameter)
    agent = LLMAgent(**agent_config)

    # Create code executor with simulator instance
    code_executor = CodeExecutor(simulator)

    # Initialize simulator and get initial state
    env_feedback = simulator.initailize_simulator()
    execution_result = None

    # Initialize control variables
    should_exit = False
    last_key_time = time.time()
    key_pressed = False

    listener = keyboard.Listener(on_press=handle_key_press)
    listener.daemon = False  # Not a daemon - we want main to exit if listener exits
    listener.start()

    print("Press 'Esc' to exit")
    print("Press 'c' to continue (or wait 1 second for auto-continue)")

    # Monitor the listener thread and implement auto-continue
    try:
        while listener.is_alive() and not should_exit:
            current_time = time.time()

            # Check if 1 second has passed since last key press
            if current_time - last_key_time >= 1.0:
                print("Auto-continuing after 1 second...")
                last_key_time = current_time

                if not process_ai_step():
                    should_exit = True
                    break

            time.sleep(0.1)  # Check every 0.1 seconds for responsiveness

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        listener.stop()


if __name__ == "__main__":
    main()
