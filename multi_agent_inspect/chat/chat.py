from dotenv import load_dotenv
import asyncio
import httpx

load_dotenv(override=True)

client = httpx.AsyncClient()
URL = "http://localhost:8000/gpt"


async def get_structured_json_response_from_gpt(
    messages, response_format, model="gpt-4o-mini", temperature=0.5, retry=0
) -> dict:

    payload = {
        "messages": messages,
        "response_format": response_format,
        "model": model,
        "temperature": temperature,
    }

    response = await client.post(URL, json=payload, timeout=None)

    data = response.json()["result"]

    return data


async def main():
    response = await get_structured_json_response_from_gpt(
        messages=[
            {
                "role": "system",
                "content": "Please think step by step and then solve the task.",
            },
            {
                "role": "user",
                "content": "What is the captial of France? A: Paris B: London C: Berlin D: Madrid.",
            },
        ],
        response_format={
            "thinking": "Your step by step thinking.",
            "answer": "A single letter, A, B, C or D.",
        },
    )
    print(response)


if __name__ == "__main__":

    asyncio.run(main())
