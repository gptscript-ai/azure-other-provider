import json
import os
import subprocess
import sys
from dataclasses import dataclass

import requests
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from openai import OpenAI

endpoint: str
api_key: str


@dataclass
class Config:
    endpoint: str
    api_key: str

    def to_json(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__,
            sort_keys=True)


async def list_resource_groups(client: ResourceManagementClient):
    group_list = client.resource_groups.list()

    column_width = 40
    print("Resource Group".ljust(column_width) + "Location", file=sys.stderr)
    print("-" * (column_width * 2), file=sys.stderr)
    for group in list(group_list):
        print(f"{group.name:<{column_width}}{group.location}", file=sys.stderr)
    print()


async def list_serverless(client: ResourceManagementClient, resource_group: str):
    filter = "resourceType eq 'Microsoft.MachineLearningServices/workspaces/serverlessEndpoints'"
    resources = client.resources.list_by_resource_group(resource_group,
                                                        filter=filter
                                                        )

    column_width = 40
    print(f"Serverless Endpoints in {resource_group}", file=sys.stderr)
    print("Workspace".ljust(column_width) + "Model Name", file=sys.stderr)
    print("-" * (column_width * 2), file=sys.stderr)
    for resource in list(resources):
        r = client.resources.get_by_id(resource_id=resource.id, api_version="2024-01-01-preview")
        model_id = r.properties["modelSettings"]["modelId"]
        last_slash_index = model_id.rfind('/')
        if last_slash_index != -1:
            model_id = model_id[last_slash_index + 1:]
        workspace = resource.name.split("/")[0]
        print(f"{workspace:<{column_width}}{model_id}", file=sys.stderr)
    print("", file=sys.stderr)


async def list_online(client: ResourceManagementClient, resource_group: str):
    filter = "resourceType eq 'Microsoft.MachineLearningServices/workspaces/onlineEndpoints'"
    resources = client.resources.list_by_resource_group(resource_group,
                                                        filter=filter
                                                        )

    column_width = 40
    print(f"Online Endpoints in {resource_group}", file=sys.stderr)
    print("Workspace".ljust(column_width) + "Model Name", file=sys.stderr)
    print("-" * (column_width * 2), file=sys.stderr)
    for resource in list(resources):
        r = client.resources.get_by_id(resource_id=resource.id, api_version="2024-01-01-preview")
        model_id = r.properties["traffic"].keys()
        model_id = list(model_id)[0]
        last_dash_index = model_id.rfind('-')
        if last_dash_index != -1:
            model_id = model_id[:last_dash_index]
        workspace = resource.name.split("/")[0]
        print(f"{workspace:<{column_width}}{model_id}", file=sys.stderr)
    print("", file=sys.stderr)


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


async def get_azure_config(model_name: str | None) -> Config | None:
    global endpoint
    global api_key

    if 'GPTSCRIPT_AZURE_ENDPOINT' in os.environ:
        endpoint = os.environ["GPTSCRIPT_AZURE_ENDPOINT"]

    if 'GPTSCRIPT_AZURE_API_KEY' in os.environ:
        api_key = os.environ["GPTSCRIPT_AZURE_API_KEY"]

    if 'endpoint' in globals() and 'api_key' in globals():
        return Config(
            endpoint=endpoint,
            api_key=api_key,
        )

    credential = DefaultAzureCredential()

    if 'AZURE_SUBSCRIPTION_ID' not in os.environ:
        print("Set AZURE_SUBSCRIPTION_ID environment variable", file=sys.stderr)
        return None
    else:
        subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]

    resource_client = ResourceManagementClient(credential=credential, subscription_id=subscription_id)
    model_id: str

    if "GPTSCRIPT_AZURE_RESOURCE_GROUP" in os.environ:
        resource_group = os.environ["GPTSCRIPT_AZURE_RESOURCE_GROUP"]
    else:
        await list_resource_groups(resource_client)
        print("Set GPTSCRIPT_AZURE_RESOURCE_GROUP environment variable", file=sys.stderr)
        return None

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
        print("Set GPTSCRIPT_AZURE_WORKSPACE environment variable", file=sys.stderr)
        return None

    if 'model_id' not in locals():
        print(f"Did not find any matches for model name {model_name}.", file=sys.stderr)
        sys.exit(1)

    api_key = await get_api_key(credential=credential, resource=selected_resource)

    return Config(
        endpoint=endpoint + "/v1",
        api_key=api_key,
    )


def client(endpoint: str, api_key: str) -> OpenAI:
    return OpenAI(
        base_url=endpoint,
        api_key=api_key,
    )


if __name__ == "__main__":
    import asyncio

    # az login
    try:
        command = ["az", "login", "--only-show-errors", "-o",
                   "none"]
        result = subprocess.run(command, stdin=None)
    except FileNotFoundError:
        print("Azure CLI not found. Please install it.", file=sys.stderr)
        sys.exit(1)
        
    if result.returncode != 0:
        print("Failed to login to Azure.", file=sys.stderr)
        sys.exit(1)

    # get model name
    command = ["gptscript", "--quiet=true", "--disable-cache", "sys.prompt",
               "{\"message\":\"Enter the name of the model:\", \"fields\":\"name\"}"]
    result = subprocess.run(command, stdin=None, stdout=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print("Failed to run sys.prompt.", file=sys.stderr)
        sys.exit(1)

    try:
        resp = json.loads(result.stdout.strip())
        model_name = resp["name"]
    except Exception as e:
        print('Failed to get model name from sys.prompt', file=sys.stderr)

    # get azure subscription id
    command = ["gptscript", "--quiet=true", "--disable-cache", "sys.prompt",
               "{\"message\":\"Enter your azure subscription id:\", \"fields\":\"id\"}"]
    result = subprocess.run(command, stdin=None, stdout=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print("Failed to run sys.prompt.", file=sys.stderr)
        sys.exit(1)
    try:
        resp = json.loads(result.stdout.strip())
        azure_subscription_id = resp["id"]
        os.environ["AZURE_SUBSCRIPTION_ID"] = azure_subscription_id
    except Exception as e:
        print('Failed to get azure subscription id from sys.prompt', file=sys.stderr)

    loop = asyncio.get_event_loop()
    config = loop.run_until_complete(get_azure_config(model_name))

    # get resource group
    command = ["gptscript", "--quiet=true", "--disable-cache", "sys.prompt",
               "{\"message\":\"Enter your azure resource group name:\", \"fields\":\"name\"}"]
    result = subprocess.run(command, stdin=None, stdout=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print("Failed to run sys.prompt.", file=sys.stderr)
        sys.exit(1)

    try:
        resp = json.loads(result.stdout.strip())
        azure_resource_group = resp["name"]
        os.environ["GPTSCRIPT_AZURE_RESOURCE_GROUP"] = azure_resource_group
    except Exception as e:
        print('Failed to get azure resource group from sys.prompt', file=sys.stderr)

    config = loop.run_until_complete(get_azure_config(model_name))

    # get workspace
    command = ["gptscript", "--quiet=true", "--disable-cache", "sys.prompt",
               "{\"message\":\"Enter your azure workspace name:\", \"fields\":\"name\"}"]
    result = subprocess.run(command, stdin=None, stdout=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print("Failed to run sys.prompt.", file=sys.stderr)
        sys.exit(1)

    try:
        resp = json.loads(result.stdout.strip())
        azure_workspace_name = resp["name"]
        os.environ["GPTSCRIPT_AZURE_WORKSPACE"] = azure_workspace_name
    except Exception as e:
        print('Failed to get azure workspace name from sys.prompt', file=sys.stderr)

    config = loop.run_until_complete(get_azure_config(model_name))

    config = {
        "env": {
            "GPTSCRIPT_AZURE_API_KEY": config.api_key,
            "GPTSCRIPT_AZURE_ENDPOINT": config.endpoint,
        }
    }

    print(json.dumps(config))
