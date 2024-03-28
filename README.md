1. You need AZURE_ENDPOINT configured to the endpoint of your model deployment if on Azure
2. You need AZURE_API_KEY configured to the API key of your model deployment on Azure

```
export AZURE_ENDPOINT=https://<your-deployment>.<your-region>.inference.ai.azure.com
export AZURE_API_KEY=<your-api-key>
```

## Usage Example

```
gptscript --default-model='mistral-large-latest from github.com/gptscript-ai/azure-provider' examples/helloworld.gpt
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
gptscript --default-model=mistral-large-latest examples/bob.gpt
```
