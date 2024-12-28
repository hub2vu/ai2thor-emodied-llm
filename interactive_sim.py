import ai2thor.controller

# Initialize the AI2-THOR controller
controller = ai2thor.controller.Controller(scene='FloorPlan1', width=800, height=600)
# controller.start(player_screen_width=800, player_screen_height=600)

# Launch an initial scene
# controller.reset('FloorPlan1')  # Load the first scene (kitchen)

print("Welcome to AI2-THOR interactive mode!")
print("Use commands like 'move', 'rotate', 'look', 'pickup', etc., to interact with the environment.")
print("Type 'help' for a list of commands and 'quit' to exit.")

# Function to print available commands
def print_help():
    print("\nAvailable Commands:")
    print(" - move [forward/backward/left/right]")
    print(" - look [up/down]")
    print(" - rotate [left/right]")
    print(" - interact [pickup/drop/open/close]")
    print(" - reset [scene] (e.g., reset FloorPlan1)")
    print(" - quit (to exit the simulator)")
    print("")

# Display the initial help message
print_help()

# Interactive loop
while True:
    try:
        # Wait for user input
        command = input("Enter your command: ").strip().lower()
        if command == "quit":
            print("Exiting AI2-THOR interactive mode.")
            break
        elif command.startswith("move"):
            direction = command.split()[-1]
            controller.step(action="MoveAhead" if direction == "forward" else
                            "MoveBack" if direction == "backward" else
                            "MoveLeft" if direction == "left" else
                            "MoveRight")
        elif command.startswith("rotate"):
            direction = command.split()[-1]
            controller.step(action="RotateLeft" if direction == "left" else "RotateRight")
        elif command.startswith("look"):
            direction = command.split()[-1]
            controller.step(action="LookUp" if direction == "up" else "LookDown")
        elif command.startswith("interact"):
            action = command.split()[-1]
            controller.step(action="PickupObject" if action == "pickup" else
                            "DropObject" if action == "drop" else
                            "OpenObject" if action == "open" else
                            "CloseObject")
        elif command.startswith("reset"):
            scene = command.split()[-1]
            controller.reset(scene)
            print(f"Scene reset to {scene}.")
        elif command == "help":
            print_help()
        else:
            print("Unknown command. Type 'help' for a list of valid commands.")
    except Exception as e:
        print(f"An error occurred: {e}")
