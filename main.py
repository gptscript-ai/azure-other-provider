import json
import os
import re
import sys
from typing import AsyncIterable

import requests
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from openai._streaming import Stream
from openai._types import NOT_GIVEN
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessage

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

    client = await get_azure_config(data["model"])
    res = client.chat.completions.create(
        model=data["model"],
        messages=data["messages"],
        tools=tools,
        tool_choice="auto",
        temperature=temperature,
        stream=stream)

    if not stream:
        return JSONResponse(content=jsonable_encoder(res))
    return StreamingResponse(convert_stream(res), media_type="application/x-ndjson")


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


async def list_resource_groups(client: ResourceManagementClient):
    group_list = client.resource_groups.list()

    column_width = 40
    print("Resource Group".ljust(column_width) + "Location")
    print("-" * (column_width * 2))
    for group in list(group_list):
        print(f"{group.name:<{column_width}}{group.location}")
    print()


async def list_serverless(client: ResourceManagementClient, resource_group: str):
    filter = "resourceType eq 'Microsoft.MachineLearningServices/workspaces/serverlessEndpoints'"
    resources = client.resources.list_by_resource_group(resource_group,
                                                        filter=filter
                                                        )

    column_width = 40
    print(f"Serverless Endpoints in {resource_group}")
    print("Workspace".ljust(column_width) + "Model Name")
    print("-" * (column_width * 2))
    for resource in list(resources):
        r = client.resources.get_by_id(resource_id=resource.id, api_version="2024-01-01-preview")
        model_id = r.properties["modelSettings"]["modelId"]
        last_slash_index = model_id.rfind('/')
        if last_slash_index != -1:
            model_id = model_id[last_slash_index + 1:]
        workspace = resource.name.split("/")[0]
        print(f"{workspace:<{column_width}}{model_id}")
    print()


async def list_online(client: ResourceManagementClient, resource_group: str):
    filter = "resourceType eq 'Microsoft.MachineLearningServices/workspaces/onlineEndpoints'"
    resources = client.resources.list_by_resource_group(resource_group,
                                                        filter=filter
                                                        )

    column_width = 40
    print(f"Online Endpoints in {resource_group}")
    print("Workspace".ljust(column_width) + "Model Name")
    print("-" * (column_width * 2))
    for resource in list(resources):
        r = client.resources.get_by_id(resource_id=resource.id, api_version="2024-01-01-preview")
        model_id = r.properties["traffic"].keys()
        model_id = list(model_id)[0]
        last_dash_index = model_id.rfind('-')
        if last_dash_index != -1:
            model_id = model_id[:last_dash_index]
        workspace = resource.name.split("/")[0]
        print(f"{workspace:<{column_width}}{model_id}")
    print()


async def get_api_key(credential, resource) -> str:
    token = credential.get_token("https://management.azure.com/.default")
    headers = {
        "Authorization": f"Bearer {token.token}",
        "Content-Type": "application/json"
    }
    response = requests.post(
        f"https://management.azure.com{resource.id}/listKeys?api-version=2024-01-01-preview",
        headers=headers)
    return response.json()["primaryKey"]


async def get_azure_config(model_name: str | None) -> OpenAI:
    credential = DefaultAzureCredential()
    if 'AZURE_SUBSCRIPTION_ID' not in os.environ:
        print("Set AZURE_SUBSCRIPTION_ID environment variable")
        sys.exit(1)
    else:
        subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]

    resource_client = ResourceManagementClient(credential=credential, subscription_id=subscription_id)
    model_id: str
    endpoint: str
    api_key: str

    if "GPTSCRIPT_AZURE_RESOURCE_GROUP" in os.environ:
        resource_group = os.environ["GPTSCRIPT_AZURE_RESOURCE_GROUP"]
    else:
        await list_resource_groups(resource_client)
        print("Set GPTSCRIPT_AZURE_RESOURCE_GROUP environment variable")
        sys.exit(0)

    if "GPTSCRIPT_AZURE_WORKSPACE" in os.environ and model_name != None:
        filter = "resourceType eq 'Microsoft.MachineLearningServices/workspaces/onlineEndpoints' or resourceType eq 'Microsoft.MachineLearningServices/workspaces/serverlessEndpoints'"
        resources = resource_client.resources.list_by_resource_group(resource_group,
                                                                     filter=filter
                                                                     )
        for resource in list(resources):
            selected_resource = resource
            r = resource_client.resources.get_by_id(resource_id=resource.id, api_version="2024-01-01-preview")
            if r.type == "Microsoft.MachineLearningServices/workspaces/serverlessEndpoints":
                model_id = r.properties["modelSettings"]["modelId"]
                last_slash_index = model_id.rfind('/')
                if last_slash_index != -1:
                    model_id = model_id[last_slash_index + 1:]
                if model_id == model_name:
                    endpoint = r.properties["inferenceEndpoint"]["uri"]
                    break
            elif r.type == "Microsoft.MachineLearningServices/workspaces/onlineEndpoints":
                model_id = r.properties["traffic"].keys()
                model_id = list(model_id)[0]
                last_dash_index = model_id.rfind('-')
                if last_dash_index != -1:
                    model_id = model_id[:last_dash_index]
                if model_id == model_name:
                    endpoint = r.properties["scoringUri"]
                    last_slash_index = endpoint.rfind('/')
                    if last_slash_index != -1:
                        endpoint = endpoint[:last_slash_index]
                    break

    else:
        await list_serverless(resource_client, resource_group)
        await list_online(resource_client, resource_group)
        print("Set GPTSCRIPT_AZURE_WORKSPACE environment variables")
        sys.exit(0)

    if 'model_id' not in locals():
        print(f"Did not find any matches for model name {model_name}.")
        sys.exit(1)

    api_key = await get_api_key(credential=credential, resource=selected_resource)
    client = OpenAI(
        base_url=endpoint + "/v1",
        api_key=api_key,
    )

    return client


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=int(os.environ.get("PORT", "8000")),
                log_level="debug" if debug else "critical", reload=debug, access_log=debug)
