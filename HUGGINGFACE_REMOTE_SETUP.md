# HuggingFace Remote Backend Setup

This guide explains how to use the remote HuggingFace Inference API for vision-language models like Qwen2.5-VL and Qwen3-VL.

## Why Use Remote HuggingFace?

- **Vision-Language Models**: Supports multimodal models like Qwen2.5-VL, Qwen3-VL, LLaVA, etc.
- **No Local GPU Required**: Runs on HuggingFace's infrastructure
- **Easy Setup**: Just need an API token
- **Cost-Effective**: Free tier available, pay-per-use for Pro

## Setup Instructions

### 1. Get HuggingFace API Token

1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Copy the token

### 2. Set Environment Variable

```bash
export HUGGINGFACEHUB_API_TOKEN="hf_your_token_here"
```

Or add to your `~/.bashrc` or `~/.zshrc`:
```bash
echo 'export HUGGINGFACEHUB_API_TOKEN="hf_your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

### 3. Configure Your YAML

Use the `huggingface_remote` backend in your config file:

```yaml
llm_config:
  backend: 'huggingface_remote'
  model: 'Qwen/Qwen2.5-VL-8B-Instruct'
  model_kwargs:
    max_new_tokens: 512
    temperature: 0.1
    do_sample: true
    timeout: 120
```

## Supported Models

Vision-language models that work with the remote backend:

- `Qwen/Qwen2.5-VL-7B-Instruct`
- `Qwen/Qwen2.5-VL-8B-Instruct`
- `Qwen/Qwen3-VL-8B-Instruct` (if available on HF Inference API)
- `llava-hf/llava-1.5-7b-hf`
- `microsoft/Phi-3-vision-128k-instruct`
- And many more...

**Note**: Some models may require Pro subscription or may not be available on the free Inference API.

## Backend Comparison

| Backend | Local/Remote | Supports Vision | GPU Required | Cost |
|---------|--------------|-----------------|--------------|------|
| `huggingface` | Local | No | Yes | Free (hardware cost) |
| `huggingface_remote` | Remote | Yes | No | Free tier + pay-per-use |
| `together` | Remote | Yes | No | Pay-per-use |
| `openai` | Remote | Yes | No | Pay-per-use |

## Troubleshooting

### "Model not found" error
- Check if the model is available on HuggingFace Inference API
- Some models are only available with Pro subscription

### Timeout errors
- Increase `timeout` in `model_kwargs`
- Try a smaller model
- Check your internet connection

### Rate limiting
- Free tier has rate limits
- Consider upgrading to HuggingFace Pro
- Or use a different backend like Together AI

## Example Usage

See `config/sample_task_1_hugging_face_QWEN_8B.yaml` for a complete example configuration.

Run with:
```bash
python main.py --config config/sample_task_1_hugging_face_QWEN_8B.yaml
```
