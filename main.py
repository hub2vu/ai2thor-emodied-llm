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


def handle_key_press(
        key: keyboard.KeyCode):
    """
    Handles the key press events for 'esc' and 'c'.
    """
    global env_feedback, execution_result
    global agent, simulator, code_executor, ai_message
    if key == keyboard.Key.esc:  # Exit if the 'Esc' key is pressed
        print("Exiting...")
        return False  # Stop the listener
    elif hasattr(key, 'char') and key.char == 'c':  # Check for 'c' key
        print("Continuing...")

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

    elif hasattr(key, 'char'):  # Handle other key presses
        print(f"Wrong key press: {key.char}, Skipping...")
        print("Press 'Esc' to exit")
        print("Press 'c' to continue")


def main() -> None:
    """Main function to run the script."""
    # I had to do that to make them accessible from the listener function,
    # I might have used a class
    # and stored them in the class state instead
    global env_feedback, execution_result
    global agent, simulator, code_executor, ai_message

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
    # while True:
    #     print("press esc to exit")
    #     print("press c to continue")
    #     event = keyboard.read_event()  # Wait for a keyboard event
    #     if event.event_type == "down":  # Only capture key down events
    #         print(f"Key pressed: {event.name}")
    #         if event.name == "esc":  # Exit if the 'Esc' key is pressed
    #             print("Exiting...")
    #             break
    #         elif event.name == "c":
    #             print("Continuing...")
    #             ai_message = agent.send_environment_feedback(
    #                 env_feedback, return_msg_id)
    #             # make the simulator execute the agent action in temp_state
    #             temp_env_feedback, return_msg_id = simulator.execute_action(
    #                 ai_message)
    #             # print the first env feedback and the agent message
    #             print(f"Env Feedback: {env_feedback}")
    #             print(f"AI Message: {ai_message}")
    #             print("--------------------------------------------------")
    #             # put the temp state in the env_feedback
    #             env_feedback = temp_env_feedback
    #         else:
    #             print(f"wrong key press: {event.name}, Skipping...")

    listener = keyboard.Listener(on_press=handle_key_press)
    listener.daemon = False  # Not a daemon - we want main to exit if listener exits
    listener.start()

    # Monitor the listener thread - exit if it stops running
    try:
        while listener.is_alive():
            listener.join(0.5) # Check every 0.5 seconds
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        listener.stop()


if __name__ == "__main__":
    main()
