import json
import os
import re
from typing import AsyncIterable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from openai._streaming import Stream
from openai.types.chat import ChatCompletionChunk

debug = True

if "AZURE_API_KEY" in os.environ:
    api_key = os.environ["AZURE_API_KEY"]
else:
    raise SystemExit("AZURE_API_KEY not found in environment variables")

if "AZURE_ENDPOINT" in os.environ:
    endpoint = os.environ["AZURE_ENDPOINT"]
else:
    raise SystemExit("AZURE_ENDPOINT is required")

app = FastAPI()
client = OpenAI(base_url=endpoint + "/v1", api_key=api_key)


def log(*args):
    if debug:
        print(*args)


@app.middleware("http")
async def log_body(request: Request, call_next):
    body = await request.body()
    log("REQUEST BODY: ", body)
    return await call_next(request)


@app.get("/")
async def get_root():
    return "ok"


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    response = json.loads(client.models.list().json())
    return JSONResponse(content=response)


@app.post("/v1/chat/completions")
async def oai_post(request: Request):
    data = await request.body()
    data = json.loads(data)

    try:
        tools = data["tools"]
    except Exception as e:
        log("No tools provided: ", e)
        tools = None

    model = data["model"]

    try:
        messages = data["messages"]
        for message in messages:
            if 'content' in message.keys() and message["content"].startswith("[TOOL_CALLS] "):
                message["content"] = ""

            if 'tool_call_id' in message.keys():
                message["name"] = re.sub(r'^call_(.*)_\d$', r'\1', message["tool_call_id"])
    except Exception as e:
        log("an error happened mapping tool_calls/tool_call_ids: ", e)
        messages = None

    res = client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="auto",
                                         stream=True)
    return StreamingResponse(convert_stream(res), media_type="application/x-ndjson")


async def convert_stream(stream: Stream[ChatCompletionChunk]) -> AsyncIterable[str]:
    for chunk in stream:
        log("CHUNK: ", chunk.json())
        yield "data: " + str(chunk.json()) + "\n\n"


if __name__ == "__main__":
    import uvicorn
    debug = os.environ.get("GPTSCRIPT_DEBUG", "false") == "true"
    uvicorn.run("main:app", host="127.0.0.1", port=int(os.environ.get("PORT", "8000")),
                log_level="debug" if debug else "critical", reload=debug, access_log=debug)
