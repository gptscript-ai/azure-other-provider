1. You need MISTRAL_ENDPOINT configured to the endpoint of your mistral deployment if using Azure
2. You need MISTRAL_API_KEY configured to the API key of your mistral deployment if using Azure, or to your La
   Plateforme API key.

```
export MISTRAL_ENDPOINT=https://<your-deployment>.<your-region>.inference.ai.azure.com
export MISTRAL_API_KEY=<your-api-key>
```

## Usage Example

```
gptscript --default-model='mistral-large-latest from github.com/gptscript-ai/mistral-openai-shim' examples/helloworld.gpt
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
export OPENAI_BASE_URL=http://127.0.0.1:8000
gptscript --default-model=mistral-large-latest examples/bob.gpt
```