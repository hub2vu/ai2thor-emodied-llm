# AI2thor-Emodied-LLM

## Description

- This is a simple project that uses LLMs to control a home robot in a simulated environment.
- The robot is tasked to do simple cooking tasks in a simulated home environment.
- The project uses AI2Thor simulator and an iThor environment.
- The project supports multiple LLM backends including self-hosted Ollama, OpenAI, Together AI, and Hugging Face models.

## Installation

```bash
# Create the venv
python -m venv .venv
# Activate the venv
source .venv/bin/activate
# Install the requirements
pip install -r requirements.txt
```

## Running

### Using Self-Hosted Ollama (Default)

The project is configured by default to use a self-hosted Ollama instance with the Gemma3:12b model:

```bash
# Source the virtual env
source .venv/bin/activate
# Run with Ollama backend (no API key needed)
python main.py --config-file config/sample_task_1_ollama.yaml
```

### Using OpenAI

```bash
# Source the virtual env
source .venv/bin/activate
# Add your Open AI Token
export OPENAI_API_KEY=YOUR_OPEN_AI_PROJECT_TOKEN
# Run the code
python main.py --config-file config/sample_task_1.yaml
```

## Supported LLM Backends

The project supports multiple LLM backends that can be configured in the YAML config file:

- **ollama**: Self-hosted Ollama instance (recommended for local deployment)
  - Requires Ollama running on your machine or network
  - Configure `base_url` in `model_kwargs` to point to your Ollama server
  - Example: `base_url: 'http://192.168.1.77:11434'`

- **openai**: OpenAI API (GPT-4o, GPT-4, etc.)
  - Requires `OPENAI_API_KEY` environment variable

- **together**: Together AI API
  - Requires Together AI API key

- **huggingface**: Local Hugging Face models
  - Runs models locally on your machine
  - Requires GPU for optimal performance

- **huggingface_remote**: Hugging Face Inference API
  - Uses Hugging Face's hosted inference endpoints

To switch backends, modify the `backend` field in your config file (e.g., `config/sample_task_1_ollama.yaml`).

## Architecture

### How the Simulator and LLM Interact

The project implements a perception-action loop where the LLM acts as the "brain" of the robot, making decisions based on visual input and environment state:

#### Main Components

1. **SimulatorBackend** (`src/simulator.py`)
   - Wraps the AI2Thor simulator
   - Provides robot control methods (move, rotate, pick, put, toggle, etc.)
   - Captures camera images and environment state after each action
   - Returns structured feedback including agent position, rotation, visible objects, and success status

2. **LLMAgent** (`src/llm_agent.py`)
   - Manages conversation with the LLM
   - Sends multimodal messages combining:
     - Environment state (position, rotation, success/failure)
     - List of visible objects with their IDs
     - Screenshot from the robot's camera
   - Maintains conversation history for context
   - Supports multiple backends (Ollama, OpenAI, Together AI, Hugging Face)

3. **CodeExecutor** (`src/code_executor.py`)
   - Parses Python code from LLM responses
   - Executes code in a controlled environment
   - Returns execution results and updated environment state

#### Execution Loop

The main execution loop (`main.py`) follows this cycle:

1. **Initialize**: Simulator starts and captures initial environment state
2. **Send to LLM**: Environment feedback (state + visible objects + screenshot) is sent to the LLM
3. **LLM Reasoning**: The LLM analyzes the visual input and environment state, then generates Python code
4. **Code Execution**: The generated code is parsed and executed against the simulator
5. **Feedback**: The simulator returns the new environment state
6. **Repeat**: Loop continues until `simulator.done()` is called or user exits

This creates a closed-loop system where the LLM continuously perceives the environment through vision and makes decisions by generating executable code.

## Controls

While the program is running, you can control execution using keyboard commands:

- **`c`** - Continue to next step (manually trigger the next AI action)
- **`Esc`** - Exit the program immediately

The program also features **auto-continue** functionality: if no key is pressed, the program will automatically continue to the next step after 1 second. This allows the robot to operate continuously without manual intervention while still giving you the option to pause execution when needed.

## Demo

This is a small demo video to demonstrate the project

<a href="https://www.youtube.com/watch?v=AZHI45pB9r4" target="_blank">
  <img src="https://img.youtube.com/vi/AZHI45pB9r4/maxresdefault.jpg" alt="Video Thumbnail" style="width:100%;max-width:640px;">
</a>
