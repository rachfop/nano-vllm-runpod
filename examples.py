#!/usr/bin/env python3
"""
Example usage of nano-vLLM Runpod deployment.
This demonstrates how to interact with the deployed service.
"""

import asyncio
import json

import aiohttp
import requests

# Example API endpoints (replace with your actual Runpod endpoint)
RUNPOD_ENDPOINT = "https://api.runpod.ai/v2/your-endpoint-id"
RUNPOD_API_KEY = "your-api-key-here"


def basic_text_generation():
    """Basic text generation example"""
    print("=== Basic Text Generation ===")

    payload = {
        "input": {
            "prompt": "What is artificial intelligence?",
            "max_tokens": 100,
            "temperature": 0.7,
        }
    }

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(f"{RUNPOD_ENDPOINT}/run", json=payload, headers=headers)

    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result.get('output', {}).get('text', 'No text generated')}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def openai_compatible_format():
    """OpenAI-compatible format example"""
    print("\n=== OpenAI-Compatible Format ===")

    payload = {
        "input": {
            "prompt": "Explain quantum computing in simple terms:",
            "max_tokens": 150,
            "temperature": 0.8,
            "openai_route": True,
        }
    }

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(f"{RUNPOD_ENDPOINT}/run", json=payload, headers=headers)

    if response.status_code == 200:
        result = response.json()
        output = result.get("output", {})
        if isinstance(output, dict) and "choices" in output:
            text = output["choices"][0].get("text", "")
            print(f"Response: {text}")
        else:
            print(f"Response: {output}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


async def streaming_example():
    """Streaming response example"""
    print("\n=== Streaming Example ===")

    payload = {
        "input": {
            "prompt": "Tell me a short story about space exploration:",
            "max_tokens": 200,
            "temperature": 0.9,
            "stream": True,
        }
    }

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }

    # Note: This is a simplified example. Real streaming would require
    # handling the streaming response properly
    response = requests.post(f"{RUNPOD_ENDPOINT}/run", json=payload, headers=headers)

    if response.status_code == 200:
        result = response.json()
        print(f"Streaming response: {result}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def batch_processing():
    """Batch processing multiple prompts"""
    print("\n=== Batch Processing ===")

    prompts = [
        "What is machine learning?",
        "Explain neural networks",
        "What are transformers in AI?",
    ]

    for i, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {i+1}: {prompt}")

        payload = {"input": {"prompt": prompt, "max_tokens": 80, "temperature": 0.7}}

        headers = {
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            f"{RUNPOD_ENDPOINT}/run", json=payload, headers=headers
        )

        if response.status_code == 200:
            result = response.json()
            print(
                f"Response: {result.get('output', {}).get('text', 'No text generated')}"
            )
        else:
            print(f"Error: {response.status_code} - {response.text}")


def main():
    """Run all examples"""
    print("nano-vLLM Runpod Deployment Examples")
    print("=" * 40)

    # Check if API key is set
    if RUNPOD_API_KEY == "your-api-key-here":
        print("⚠️  Please set your Runpod API key and endpoint URL")
        print("Edit this file and update RUNPOD_ENDPOINT and RUNPOD_API_KEY")
        return

    try:
        basic_text_generation()
        openai_compatible_format()

        # Run streaming example
        asyncio.run(streaming_example())

        batch_processing()

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Network error: {e}")
        print("Make sure your Runpod endpoint is running and accessible")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
