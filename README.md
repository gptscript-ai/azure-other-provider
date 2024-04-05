1. You must be authenticated with the Azure CLI
2. You need the env variable `AZURE_SUBSCRIPTION_ID` to be configured
3. You need the env variable `GPTSCRIPT_AZURE_RESOURCE_GROUP` to be configured
4. You need the env variable `GPTSCRIPT_AZURE_WORKSPACE` to be configured

```
az login
export AZURE_SUBSCRIPTION_ID=<your-subscription-key>
export GPTSCRIPT_AZURE_RESOURCE_GROUP=<your-resource-group>
export GPTSCRIPT_AZURE_WORKSPACE=<your-workspace>
```

## Usage Example

```
gptscript --default-model='Mistral-large from github.com/gptscript-ai/azure-other-provider' examples/helloworld.gpt
```

## Development

Run using the following commands

```
python -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt
./run.sh
```

```
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export GPTSCRIPT_DEBUG=true
gptscript --default-model=Mistral-large examples/bob.gpt
```
