"""Test script to validate HuggingFace remote VLM support.

This script tests whether the huggingface_remote backend properly handles
multimodal (image + text) inputs for Vision-Language Models.
"""

import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np

# Test 1: Using LangChain's ChatHuggingFace + HuggingFaceEndpoint
def test_langchain_huggingface_vlm():
    """Test current implementation using LangChain wrappers."""
    print("=" * 80)
    print("TEST 1: LangChain ChatHuggingFace + HuggingFaceEndpoint")
    print("=" * 80)

    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        from langchain_core.messages import HumanMessage, SystemMessage

        # Check for API token
        api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not api_token:
            print("❌ HUGGINGFACEHUB_API_TOKEN not set!")
            print("   Please run: export HUGGINGFACEHUB_API_TOKEN='hf_your_token'")
            return False

        print(f"✓ API Token found: {api_token[:10]}...")

        # Create a simple test image (red square)
        img = Image.new('RGB', (100, 100), color='red')
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        img_data_url = f"data:image/png;base64,{img_base64}"

        print("✓ Created test image (100x100 red square)")

        # Initialize HuggingFace endpoint
        print("\nInitializing HuggingFace endpoint...")
        hf_endpoint = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-VL-8B-Instruct",
            task="text-generation",
            max_new_tokens=100,
            temperature=0.1,
        )
        print("✓ HuggingFaceEndpoint created")

        # Wrap with ChatHuggingFace
        llm = ChatHuggingFace(llm=hf_endpoint)
        print("✓ ChatHuggingFace wrapper created")

        # Create multimodal message (OpenAI style)
        print("\nSending multimodal message...")
        messages = [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content=[
                {"type": "text", "text": "What color is this image?"},
                {"type": "image_url", "image_url": {"url": img_data_url}}
            ])
        ]

        print(f"  - System message: 'You are a helpful AI assistant.'")
        print(f"  - Human message with text and image")
        print(f"  - Image size: {len(img_data_url)} characters")

        # Try to invoke
        try:
            response = llm.invoke(messages)
            print("\n✅ SUCCESS! Response received:")
            print(f"   {response.content[:200]}...")
            return True
        except Exception as e:
            print(f"\n❌ FAILED during invoke:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)[:300]}")
            return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        return False


# Test 2: Using HuggingFace InferenceClient directly
def test_huggingface_inference_client():
    """Test using HuggingFace InferenceClient directly."""
    print("\n" + "=" * 80)
    print("TEST 2: HuggingFace InferenceClient (Direct)")
    print("=" * 80)

    try:
        from huggingface_hub import InferenceClient

        # Check for API token
        api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not api_token:
            print("❌ HUGGINGFACEHUB_API_TOKEN not set!")
            return False

        print(f"✓ API Token found: {api_token[:10]}...")

        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        img_data_url = f"data:image/png;base64,{img_base64}"

        print("✓ Created test image (100x100 red square)")

        # Initialize InferenceClient
        print("\nInitializing InferenceClient...")
        client = InferenceClient(token=api_token)
        print("✓ InferenceClient created")

        # Try chat completion with image
        print("\nSending chat completion request with image...")

        try:
            # Method 1: Using chat_completion with image
            response = client.chat_completion(
                model="Qwen/Qwen2.5-VL-8B-Instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What color is this image?"},
                            {"type": "image_url", "image_url": {"url": img_data_url}}
                        ]
                    }
                ],
                max_tokens=100,
            )

            print("\n✅ SUCCESS! Response received:")
            print(f"   {response.choices[0].message.content[:200]}...")
            return True

        except Exception as e:
            print(f"\n❌ FAILED during chat_completion:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)[:500]}")

            # Try alternative format
            print("\n   Trying alternative image format...")
            try:
                response = client.chat_completion(
                    model="Qwen/Qwen2.5-VL-8B-Instruct",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What color is this image?"},
                                {"type": "image", "image": img_base64}
                            ]
                        }
                    ],
                    max_tokens=100,
                )
                print("\n✅ SUCCESS with alternative format! Response received:")
                print(f"   {response.choices[0].message.content[:200]}...")
                return True
            except Exception as e2:
                print(f"\n❌ Also failed with alternative format:")
                print(f"   Error: {str(e2)[:300]}")
                return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Try: pip install huggingface-hub")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        return False


def main():
    """Run all tests."""
    print("\n🧪 Testing HuggingFace Remote VLM Support\n")

    # Check prerequisites
    print("Checking prerequisites...")
    api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not api_token:
        print("\n⚠️  HUGGINGFACEHUB_API_TOKEN environment variable not set!")
        print("Please set it before running tests:")
        print("  export HUGGINGFACEHUB_API_TOKEN='hf_your_token_here'")
        print("\nGet your token at: https://huggingface.co/settings/tokens\n")
        return

    # Run tests
    test1_result = test_langchain_huggingface_vlm()
    test2_result = test_huggingface_inference_client()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Test 1 (LangChain ChatHuggingFace): {'✅ PASSED' if test1_result else '❌ FAILED'}")
    print(f"Test 2 (HuggingFace InferenceClient): {'✅ PASSED' if test2_result else '❌ FAILED'}")

    if test1_result:
        print("\n✅ The current huggingface_remote backend should work!")
        print("   No changes needed to llm_agent.py")
    elif test2_result:
        print("\n⚠️  LangChain wrapper doesn't work, but InferenceClient does!")
        print("   We should implement a new backend using InferenceClient directly")
    else:
        print("\n❌ Both approaches failed!")
        print("   This could be due to:")
        print("   - Model not available on Inference API")
        print("   - API limitations or rate limiting")
        print("   - Incorrect image format")
        print("   - Model doesn't support chat completion API")

    print("\n")


if __name__ == "__main__":
    main()
