import json
import os
from typing import AsyncIterable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from openai._streaming import Stream
from openai.types.chat import ChatCompletionChunk

api_key = os.environ["MISTRAL_API_KEY"]
endpoint = os.environ["MISTRAL_ENDPOINT"]
app = FastAPI()
client = OpenAI(base_url=endpoint + "/v1", api_key=api_key)


# @app.middleware("http")
# async def log_body(request: Request, call_next):
#     body = await request.body()
#     print("REQUEST BODY: ", body)
#     return await call_next(request)


@app.get("/models")
async def list_models() -> JSONResponse:
    return JSONResponse(content={"data": [{"id": "azureai", "name": "Azure AI"}]})


@app.post("/chat/completions")
async def oai_post(request: Request):
    data = await request.body()
    data = json.loads(data)
    try:
        tools = data["tools"]
    except Exception as e:
        print("an error happened with tools: ", e)
        tools = None

    try:
        messages = data["messages"]
        for message in messages:
            if message["content"] is not None and message["content"].startswith("[TOOL_CALLS] "):
                message["content"] = ""
            # This is a weird gptscript thing i think
            if message["role"] == "tool" or message["role"] == "model":
                message["role"] = "assistant"
    except Exception as e:
        print("an error happened: ", e)
        messages = None

    res = client.chat.completions.create(model="azureai", messages=messages, tools=tools, tool_choice="auto",
                                         stream=True)

    return StreamingResponse(convert_stream(res), media_type="application/x-ndjson")


async def convert_stream(stream: Stream[ChatCompletionChunk]) -> AsyncIterable[str]:
    for chunk in stream:
        yield "data: " + str(chunk.json()) + "\n\n"
