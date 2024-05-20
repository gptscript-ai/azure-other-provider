import json
import os
import re
from typing import AsyncIterable

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from openai import Stream
from openai._types import NOT_GIVEN
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage

import helpers

debug = os.environ.get("GPTSCRIPT_DEBUG", "false") == "true"


def log(*args):
    if debug:
        print(*args)


app = FastAPI()


@app.middleware("http")
async def log_body(request: Request, call_next):
    body = await request.body()
    log("REQUEST BODY: ", body)
    return await call_next(request)


@app.get("/")
async def get_root():
    return "ok"


# Only needed when running standalone. With GPTScript, the `id` returned by this endpoint must match the model you are passing in.
@app.get("/v1/models")
async def list_models() -> JSONResponse:
    return JSONResponse(content={"data": [{"id": "Mistral-large", "name": "Your model"}]})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.body()
    data = json.loads(data)

    try:
        tools = data["tools"]
    except Exception as e:
        log("No tools provided: ", e)
        tools = []

    model = data['model']
    try:
        messages = data["messages"]
        for message in messages:
            if 'content' in message.keys() and message["content"].startswith("[TOOL_CALLS] "):
                message["content"] = ""

            if 'tool_call_id' in message.keys():
                message["name"] = re.sub(r'^call_(.*)_\d$', r'\1', message["tool_call_id"])

            if 'role' in message.keys() and message['role'] == 'assitant':
                message = ChatCompletionMessage.model_validate(message)

    except Exception as e:
        log("an error happened mapping tool_calls/tool_call_ids: ", e)
        messages = None

    temperature = data.get("temperature", NOT_GIVEN)
    if temperature is not NOT_GIVEN:
        temperature = float(temperature)

    stream = data.get("stream", False)

    config = await helpers.get_azure_config(data["model"])
    if config == None:
        raise HTTPException(status_code=400,
                            detail="Azure config not found. Please ensure you have configured the environment variables correctly.")

    client = helpers.client(
        endpoint=config.endpoint,
        api_key=config.api_key,
    )

    try:
        res: Stream[ChatCompletionChunk] | ChatCompletion = client.chat.completions.create(
            model=data["model"],
            messages=data["messages"],
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
            stream=stream)

        if not stream:
            return JSONResponse(content=jsonable_encoder(res))

        return StreamingResponse(convert_stream(res), media_type="application/x-ndjson")
    except Exception as e:
        try:
            log("Error occurred: ", e.__dict__)
            error_code = e.status_code
            error_message = e.message
        except:
            error_code = 500
            error_message = str(e)
        raise HTTPException(status_code=error_code, detail=f"Error occurred: {error_message}")


async def convert_stream(stream: Stream[ChatCompletionChunk]) -> AsyncIterable[str]:
    for chunk in stream:
        for choice in chunk.choices:
            if choice.delta.tool_calls is None:
                continue

            for tool_call in choice.delta.tool_calls:
                tool_call.index = tool_call.index or 0
                tool_call.type = tool_call.type or 'function'
                tool_call.id = f"call_{tool_call.function.name}_{tool_call.index}"

        log("CHUNK: ", chunk.model_dump_json())

        yield "data: " + str(chunk.model_dump_json()) + "\n\n"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=int(os.environ.get("PORT", "8000")),
                log_level="debug" if debug else "critical", reload=debug, access_log=debug)
