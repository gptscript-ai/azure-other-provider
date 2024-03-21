import json
import os
import re
from typing import AsyncIterable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from openai._streaming import Stream
from openai.types.chat import ChatCompletionChunk

if "MISTRAL_API_KEY" in os.environ:
    api_key = os.environ["MISTRAL_API_KEY"]
else:
    raise SystemExit("MISTRAL_API_KEY not found in environment variables")

if "MISTRAL_ENDPOINT" in os.environ:
    endpoint = os.environ["MISTRAL_ENDPOINT"]
else:
    raise SystemExit("MISTRAL_ENDPOINT not found in environment variables")

# class BackgroundTasks(threading.Thread):
#     async def run(self, *args, **kwargs):
#         while True:
#             try:
#                 for _ in sys.stdin:
#                     pass
#             except:
#                 "Closing background thread."
#                 SystemExit(0)
#
#
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     t = BackgroundTasks()
#     t.daemon = True
#     t.start()
#
#     yield
#
#
# app = FastAPI(lifespan=lifespan)
app = FastAPI()
client = OpenAI(base_url=endpoint + "/v1", api_key=api_key)


@app.middleware("http")
async def log_body(request: Request, call_next):
    body = await request.body()
    print("REQUEST BODY: ", body)
    return await call_next(request)


@app.post("/")
async def post_root():
    return 'ok'


@app.get("/")
async def get_root():
    return 'ok'


@app.get("/v1//models")
async def list_models() -> JSONResponse:
    try:
        response = json.loads(client.models.list().json())
        return JSONResponse(content=response)
    except:
        return JSONResponse(content={"data": [{"id": "mistral-large-latest", "name": "Azure AI"}]})


@app.post("/v1/chat/completions")
async def oai_post(request: Request):
    data = await request.body()
    data = json.loads(data)

    try:
        tools = data["tools"]
    except Exception as e:
        print("No tools provided: ", e)
        tools = []

    model = data["model"]

    try:
        messages = data["messages"]
        for message in messages:
            if 'content' in message.keys() and message["content"].startswith("[TOOL_CALLS] "):
                message["content"] = ""

            if 'tool_call_id' in message.keys():
                message["name"] = re.sub(r'^call_(.*)_\d$', r'\1', message["tool_call_id"])
    except Exception as e:
        print("an error happened mapping tool_calls/tool_call_ids: ", e)
        messages = None

    res = client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="auto",
                                         stream=True, extra_body={"random_seed": data["seed"]})
    return StreamingResponse(convert_stream(res), media_type="application/x-ndjson")


async def convert_stream(stream: Stream[ChatCompletionChunk]) -> AsyncIterable[str]:
    for chunk in stream:
        print("CHUNK: ", chunk.json())
        yield "data: " + str(chunk.json()) + "\n\n"
