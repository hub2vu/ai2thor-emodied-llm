# AI2thor-Emodied-LLM

## Description

- This is a simple project that uses Chatgpt-4o to control a home robot.
- The robot is tasked to do a simple cooking task to do it in a simulated home environment.
- The project uses AI2Thor simulator and an iThor environment.
- The project uses langchains to ineract with openai gpt-4o model.

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

```bash
# Source the virtual env
source .venv/bin/activate
# Add your Open AI Token
export OPENAI_API_KEY=YOUR_OPEN_AI_PROJECT_TOKEN
# Run the code
python main.py --config-file config/sample_task_1.yaml
```

## Demo

This is a small demo video to demonstrate the project

<a href="https://www.youtube.com/watch?v=AZHI45pB9r4" target="_blank">
  <img src="https://img.youtube.com/vi/AZHI45pB9r4/maxresdefault.jpg" alt="Video Thumbnail" style="width:100%;max-width:640px;">
</a>
