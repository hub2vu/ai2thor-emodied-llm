"""Represents the main entry point of the project.
"""

from typing import Union
import argparse
import yaml
from pynput import keyboard

from src.simulator import (
    SimulatorBackend, EnvironmentState,
    QueryReturn
)
from src.llm_agent import LLMAgent


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
    global env_feedback, return_msg_id
    global agent, simulator, ai_message
    if key == keyboard.Key.esc:  # Exit if the 'Esc' key is pressed
        print("Exiting...")
        return False  # Stop the listener
    elif hasattr(key, 'char') and key.char == 'c':  # Check for 'c' key
        print("Continuing...")
        ai_message = agent.send_environment_feedback(
            env_feedback, return_msg_id)
        if len(ai_message.tool_calls) == 0:
            print("End of episode .. exiting")
            return False
        temp_env_feedback, return_msg_id = simulator.execute_action(
            ai_message)
        print(f"AI Message: {ai_message}")
        print("\n\n")
        print(f"Env Feedback: {temp_env_feedback}")
        print("--------------------------------------------------")
        # Update the environment feedback
        env_feedback = temp_env_feedback
    elif hasattr(key, 'char'):  # Handle other key presses
        print(f"Wrong key press: {key.char}, Skipping...")
        print("Press 'Esc' to exit")
        print("Press 'c' to continue")


def main() -> None:
    """Main function to run the script."""
    # I had to do that to make them accessible from the listiner function,
    # I might have used a class
    # and stored them in the class state instead
    global env_feedback, return_msg_id
    global agent, simulator, ai_message

    args = parse_args()

    # Access the config file path
    config_file = args.config_file
    config = read_yaml_file(config_file)
    simulator_config = config['simulator_config']
    simulator = SimulatorBackend(**simulator_config)
    agent_config = config['llm_config']
    agent = LLMAgent(
        tools=simulator.get_available_actions(), **agent_config)
    env_feedback = simulator.initailize_simulator()
    return_msg_id = ''
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
    with keyboard.Listener(on_press=handle_key_press) as listener:
        listener.join()


if __name__ == "__main__":
    main()
