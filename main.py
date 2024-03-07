import ast
import json
import os
from typing import AsyncIterable, List, Optional
from openai._streaming import Stream
from openai.types.chat import ChatCompletionChunk

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRouter
from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import ChatCompletionStreamResponse, ChatMessage
from pydantic import BaseModel
from openai import OpenAI

# from openai.types import ChatMessage as OpenAIChatMessage

app = FastAPI()
router = APIRouter()

api_key = os.environ["MISTRAL_API_KEY"]
endpoint = os.environ["MISTRAL_ENDPOINT"]
model = "azureai"

client = MistralAsyncClient(endpoint=endpoint, api_key=api_key)
oai_client = OpenAI(base_url=endpoint+"/v1", api_key=api_key)


@app.middleware("http")
async def log_body(request: Request, call_next):
    body = await request.body()
    print("REQUEST BODY: ", body)
    return await call_next(request)


class OAIMessage(BaseModel):
    role: str
    content: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict] | None = None


class OAIDelta(OAIMessage):
    pass


class OAIParameters(BaseModel):
    type: str | None = None
    properties: dict | None = None
    required: list[str] | None = None


class OAIFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: OAIParameters | None = None


class OAITool(BaseModel):
    type: str | str = "function"
    function: OAIFunction


class OAIToolCall(OAITool):
    id: str | None = None


class OAICompletionRequest(BaseModel):
    model: str
    messages: List[OAIMessage] | None = None
    max_tokens: int | None = None
    stream: bool | None = False
    seed: float | None = None
    tools: List[OAITool] | None = None
    tool_choice: List[OAITool] | Optional[str] | None = None
    top_k: int | None = None
    top_p: float | None = None
    temperature: float | None = None


class OAITopLogProb(BaseModel):
    token: str
    logprob: int | None = None
    bytes: List[int]


class OAIContent(BaseModel):
    token: str | None = None
    logprob: Optional[int] = None
    bytes: Optional[List[int]] = None
    top_logprobs: Optional[List[OAITopLogProb]] = None


class OAILogprobs(BaseModel):
    content: List[OAIContent] | None = None


class RespChoice(BaseModel):
    finish_reason: str | None = None
    index: int
    delta: OAIDelta
    logprobs: OAILogprobs | None = None
    # tool_calls: List[OAIToolCall] | None = None


class OAIUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class OAICompletionResponse(BaseModel):
    id: str
    choices: List[RespChoice]
    created: int | None = None
    model: str
    system_fingerprint: str | None = None
    object: str | None = None
    usage: OAIUsage | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None


@app.get("/models")
async def list_models() -> JSONResponse:
    return JSONResponse(content={"data": [{"id": "azureai", "name": "Azure AI"}]})

@app.get("/oai")
async def oai_test():
    tools = [{'type': 'function', 'function': {'name': 'bob',
                                               'description': "I'm Bob, a friendly guy.",
                                               'parameters':
                                                   {'type': 'object',
                                                    'properties':
                                                        {
                                                            'question':
                                                                {
                                                                    'description': 'The question to ask Bob.',
                                                                     'type': 'string'
                                                                }
                                                        },
                                                    'required': None
                                                    }
                                               }
              }
             ]
    tool_calls = [
        {"id": "call_bob_0", "type": "function", "function": {"name": "bob", "arguments": "{\"question\": \"How are you?\"}"}}
    ]

    messages = [
        {
            "role": "system",
         "content": "\nYou are task oriented system.\nYou receive input from a user, process the input from the given instructions, and then output the result.\nYour objective is to provide consistent and correct results.\nYou do not need to explain the steps taken, only provide the result to the given instructions.\nYou are referred to as a tool.\n\nAsk Bob how he is doing and let me know exactly what he said."
        },
        {
            "role": "user",
            "content": "--question=How are you?"
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": tool_calls
        },
        {
            "role": "assistant",
            "name": "bob",
            "content": "Thanks for asking \"how are you?\", I'm doing great fellow friendly AI tool!",
            "tool_call_id": "call_bob_0"
        }
    ]
    res = oai_client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="auto")
    return res

@app.post("/chat/completions")
async def oai_post(request:Request):
    data = await request.body()
    data = json.loads(data)
    try:
        tools = data["tools"]
    except:
        tools = None

    # tools = data["tools"]
    try:
        messages = data["messages"]
        for message in messages:
            if message["content"] is not None and message["content"].startswith("[TOOL_CALLS] "):
                message["content"] = ""
            if message["role"] == "tool" or message["role"] == "model":
                message["role"] = "assistant"
    except Exception as e:
        print ("an error happened: ", e)
        messages = None

    res = oai_client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="auto", stream=True)

    return StreamingResponse(convert_stream(res), media_type="application/x-ndjson")

async def convert_stream(stream: Stream[ChatCompletionChunk]) -> AsyncIterable[str]:
    for chunk in stream:
        yield "data: " + str(chunk.json()) + "\n\n"

@app.get("/test")
async def test():
    tools = [{'type': 'function', 'function': {'name': 'bob',
                                               'description': "I'm Bob, a friendly guy.",
                                               'parameters':
                                                   {'type': 'object',
                                                    'properties':
                                                        {
                                                            'question':
                                                                {
                                                                    'description': 'The question to ask Bob.',
                                                                     'type': 'string'
                                                                }
                                                        },
                                                    'required': None
                                                    }
                                               }
              }
             ]
    tool_calls = [
        {"index": 0, "id": "call_bob_0", "type": "function", "function": {"name": "bob", "arguments": "{\"question\": \"How are you?\"}"}}
    ]

    messages = [
        ChatMessage(role="system", content="\nYou are task oriented system.\nYou receive input from a user, process the input from the given instructions, and then output the result.\nYour objective is to provide consistent and correct results.\nYou do not need to explain the steps taken, only provide the result to the given instructions.\nYou are referred to as a tool.\n\nAsk Bob how he is doing and let me know exactly what he said."),
        ChatMessage(role="user", content="--question=How are you?"),
        ChatMessage(role="assistant", content="", tool_calls=tool_calls),
        ChatMessage(role="assistant", name="call_bob_0", content="Thanks for asking \"how are you?\", I'm doing great fellow friendly AI tool!")
                ]
    res = await client.chat(model=model, messages=messages, tools=tools, tool_choice="auto")
    return res

# @app.post("/chat/completions")
# @app.post("/chat/completions", response_model_exclude_none=True, response_model_exclude_unset=True,
#           response_model=OAICompletionResponse)
async def chat_completion(data: OAICompletionRequest) -> StreamingResponse:
    tools: list[dict] | None = None
    if data.tools is not None:
        tools = []
        for tool in data.tools:
            tools.append(tool.dict())

    print("TOOLS JSON: ", tools)
    messages = await map_oai_to_mistral_messages(data.messages)
    response = client.chat_stream(model=model,
                                  messages=messages,
                                  tools=tools,
                                  max_tokens=data.max_tokens,
                                  temperature=data.temperature,
                                  top_p=data.top_p,
                                  tool_choice="auto",
                                  random_seed=data.seed)

    resp = async_chunk(response)
    print("done with async_chunk")
    return StreamingResponse(resp, media_type="application/x-ndjson")


async def async_chunk(chunks: AsyncIterable[ChatCompletionStreamResponse]) -> AsyncIterable[str]:
    async for chunk in chunks:
        processed_chunk: OAICompletionResponse = map_mistral_to_oai_response(chunk)
        # print("PROCESSED CHUNK: ", processed_chunk.json())
        yield "data: " + processed_chunk.json() + "\n\n"


async def map_oai_to_mistral_messages(messages: List[OAIMessage]) -> List[ChatMessage]:
    mistral_messages: list[ChatMessage] = []

    for message in messages:
        role: str = message.role
        # if role == "system":
        #     role = "user"
        if role == "tool" or role == 'model':
            role = "assistant"

        message.content = message.content if message.content is not None else ""
        if message.tool_call_id is not None:
            tool_call_id = message.tool_call_id.replace("call_", "").replace("_0", "")
            mistral_messages.append(
                ChatMessage(role=role, content=message.content, name=tool_call_id, tool_calls=message.tool_calls))
        else:
            mistral_messages.append(ChatMessage(role=role, content=message.content, tool_calls=message.tool_calls))

    # print("MISTRAL MESSAGES: ", mistral_messages)
    return mistral_messages


async def dict_from_class(cls):
    return dict(
        (key, value)
        for (key, value) in cls.__dict__.items()
    )


def map_mistral_to_oai_response(response: ChatCompletionStreamResponse) -> OAICompletionResponse:
    # print("MISTRAL RESPONSES FROM INSIDE MAPPER: ", response)
    choices = []
    try:
        for choice in response.choices:
            tool_calls: list[dict] | None = None
            role = choice.delta.role
            if choice.delta.role is None:
                role = "model"

            try:
                for idx, tool_call in enumerate(choice.delta.tool_calls):
                    tool_calls = []
                    tool_calls.append({
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "index": idx,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
            except Exception as e:
                print("exception setting tools from choice.delta.tool_calls: ", e)
                tool_calls = None

            content = choice.delta.content
            if content == "":
                content = None
            try:
                if content is not None and content.startswith("[TOOL_CALLS] "):
                    # tool_calls = []
                    #
                    # for tool_call in json.loads(
                    #         content.replace(
                    #             "[TOOL_CALLS] ", ""
                    #         )
                    # ):
                    #     tool_calls.append(tool_call)
                    content = None
            except Exception as e:
                print("exception when trying to fix TOOL_CALLS: ", e)

            finish_reason = None
            try:
                finish_reason = map_finish_reason(choice.finish_reason)
            except Exception as e:
                print("exception when mapping finish reason: ", e)

            choices.append(RespChoice(
                finish_reason=finish_reason,
                index=choice.index,
                delta=OAIDelta(
                    index=choice.index,
                    role=role,
                    content=content,
                    tool_calls=tool_calls
                ),
                logprobs=None
            ))

    except Exception as e:
        print("exception when mapping choices: ", e)

    # usage: OAIUsage | None = None
    # if response.usage is not None:
    #     usage = OAIUsage(
    #         completion_tokens=response.usage.completion_tokens,
    #         prompt_tokens=response.usage.prompt_tokens,
    #         total_tokens=response.usage.total_tokens
    #     )

    return OAICompletionResponse(
        id=response.id,
        choices=choices,
        created=response.created,
        model=response.model,
        object=response.object,
        # usage=usage,
    )


def map_finish_reason(finish_reason: str) -> str:
    # openai supports 5 stop sequences - 'stop', 'length', 'function_call', 'content_filter', 'null'
    if finish_reason == "error":
        return "stop"
    # elif finish_reason == "tool_calls":
    #     return 'function_call'
    return finish_reason
