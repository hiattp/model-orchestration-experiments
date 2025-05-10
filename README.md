# Model Orchestration Experiments

These are experiments done with the aim of chaining multiple services/capabilities to determine whether final value can be achieved for complex use cases (reasoning about complex data in non-digital documents, for example).

## Setup

```sh
poetry install
```

### Environment variables
For experiments utilizing the Azure AI Document Intelligence service, you will need the following environment variables:

```sh
❯ cat .env
export AZURE_DOCUMENTINTELLIGENCE_ENDPOINT="<your Azure AI Document Intelligence service endpoint>"
export AZURE_DOCUMENTINTELLIGENCE_API_KEY="<one of your Azure AI Document Intelligence service API Keys>"
```

For experiments utilizing the Azure OpenAI Service, you will need the following environment variables:

```sh
❯ cat .env
export AZURE_OPENAI_ENDPOINT="<your Azure OpenAI Service endpoint>"
export AZURE_OPENAI_API_KEY="<one of your Azure OpenAI API Keys>"
export AZURE_OPENAI_CHAT_COMPLETION_MODEL_DEPLOYMENT_NAME="<the name of a model deployed in Azure AI Foundry that can be used for chat completion, e.g. 'gpt-4'>>"
export AZURE_OPENAI_TEXT_EMBEDDING_MODEL_DEPLOYMENT_NAME="<the name of a model deployed in Azure AI Foundry that can be used for text embeddings, e.g. 'text-embedding-3-large'>>"
```

For experiments utilizing the GoogleAI Service, you will need the following environment variables:

```sh
❯ cat .env
export GOOGLE_AI_GEMINI_API_KEY="<one of your Google AI API Keys>"
export GOOGLE_AI_GEMINI_CHAT_COMPLETION_MODEL_DEPLOYMENT_NAME="<the name of a model deployed in GoogleAI that can be used for chat completion, e.g. 'gemini-2.5-pro-preview-05-06'>>"
export GOOGLE_AI_GEMINI_TEXT_EMBEDDING_MODEL_DEPLOYMENT_NAME="<the name of a model deployed in GoogleAI that can be used for text embeddings, e.g. 'gemini-embedding-exp-03-07'>>"
```

For experiments utilizing the Google Cloud VertexAI Service, you will need the following environment variables:

```sh
❯ cat .env
export VERTEX_AI_PROJECT_ID="<the name of the Google Cloud project>"
export VERTEX_AI_REGION="<the name of the region in which the VertexAI Service is deployed>"
export VERTEX_AI_GEMINI_CHAT_COMPLETION_MODEL_DEPLOYMENT_NAME="<the name of a model deployed in VertexAI that can be used for chat completion, e.g. 'gemini-2.5-pro-preview-05-06'>>"
export VERTEX_AI_GEMINI_TEXT_EMBEDDING_MODEL_DEPLOYMENT_NAME="<the name of a model deployed in VertexAI that can be used for text embeddings, e.g. 'text-embedding-005'>>"
```

## Storing data
An internal folder structure has been set up to accommodate input and output files. Feel free to house files that you may use under `data/in`. Unless instructed otherwise, the experiments will save any output they may generate (if any) under `data/out`.

## Experiments

### Simple Chat Completion

Requirements:
- Azure OpenAI Service credentials

This is a simple experiment showcasing the completion of a predefined prompt and a given input by a Chat Completion service.

Usage:

```sh
❯ poetry run complete_chat_prompt -h
usage: complete_chat_prompt [-h] -i INPUT

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Simple input to the following chat prompt: <<input>> is the capital of
```

Example:
```sh
poetry run complete_chat_prompt -i Paris
```

### Simple Document Extraction and Analysis

Requirements:
- Azure AI Document Intelligence service credentials
- credentials for an AI Service where Chat Completion and Text Embedding models are available, out of: Azure OpenAI/GoogleAI/Google Cloud VertexAI
- [ChromaDB](https://www.trychroma.com/)

This is a simple experiment showcasing:
- the extraction of data from a PDF document using [layout-aware extraction](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/prebuilt/layout?view=doc-intel-4.0.0&tabs=rest%2Csample-code)
- the interpretation of the extracted data using a Chat Completion service

Usage:

Before progressing with the actual analysis, an instance of ChromaDB must be running on the local host. For this, you could either start it via the `poetry` virtual environment:

```sh
poetry run chroma run --path /path/to/where/chromadb/data/should/be/saved
```

Or [via a Docker Container](https://docs.trychroma.com/production/containers/docker).

This experiment assumes the ChromaDB instance listens to port 8000.

```sh
❯ poetry run analyze_document -h
usage: analyze_document [-h] -s {openai,googleai,vertexai} -f FILEPATH [-e FROM_PRIOR_EXPERIMENT_RUN]

options:
  -h, --help            show this help message and exit
  -s {openai,googleai,vertexai}, --service {openai,googleai,vertexai}
                        The AI Service to use
  -f FILEPATH, --filepath FILEPATH
                        Relative path to the original file out of which data needs to be extracted and analyzed
  -e FROM_PRIOR_EXPERIMENT_RUN, --from-prior-experiment-run FROM_PRIOR_EXPERIMENT_RUN
                        Expects the id of a prior experiment run from which to re-use extracted data
```

Examples:

- extracting and analyzing the data of an original file
```sh
poetry run analyze_document -s "openai" -f "data/in/path/to/original/file.pdf"
```

- re-using previously extracted data from an original file for analysis purposes
```sh
poetry run analyze_document -s "openai" -f "data/in/path/to/original/file.pdf" -e "id of prior experiment run"
```
