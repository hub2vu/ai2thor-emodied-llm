"""Represents the main entry point of the project.
"""

import argparse


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


def main() -> None:
    """Main function to run the script."""
    args = parse_args()

    # Access the config file path
    config_file = args.config_file

    # Add your main logic here
    print(f"Using config file: {config_file}")


if __name__ == "__main__":
    main()
