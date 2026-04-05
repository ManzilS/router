from openai import AsyncOpenAI
import asyncio

async def test_lmstudio_via_gateway():
    # We point to the local AI Gateway (which then proxies to LM Studio)
    client = AsyncOpenAI(
        api_key="sk-not-needed", 
        base_url="http://localhost:8080/v1"
    )

    print("Sending request to AI Gateway...")
    
    # Send the request using the exact LM Studio model name
    response = await client.chat.completions.create(
        model="qwen3-0.6b",
        messages=[
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a python script that prints Hello World."}
        ],
        stream=True
    )

    print("\nResponse from Qwen3:")
    async for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            
    print("\n\nAll done! The Gateway should have logged this exchange.")

if __name__ == "__main__":
    asyncio.run(test_lmstudio_via_gateway())
