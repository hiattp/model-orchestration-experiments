import os
import sys
import json

import asyncio
import semantic_kernel as sk
import argparse

from dotenv import load_dotenv

from chromadb import Client, HttpClient
from chromadb.config import Settings

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

from semantic_kernel.memory.memory_store_base import MemoryStoreBase
from semantic_kernel.connectors.memory.chroma import ChromaMemoryStore
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextCompletion,
    AzureTextEmbedding,
)
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.memory.volatile_memory_store import VolatileMemoryStore

from .utils import (
    make_experiment_run,
    end_experiment_run,
    get_experiment_run_destination,
)

EXPERIMENT_NAME = "SIMPLE_DOCUMENT_EXTRACTION_AND_ANALYSIS"
EXPERIMENT_SAVE_RESULTS = True


def _setup_document_intelligence_client() -> DocumentIntelligenceClient:
    AZURE_DOCUMENTINTELLIGENCE_ENDPOINT = os.getenv(
        "AZURE_DOCUMENTINTELLIGENCE_ENDPOINT"
    )
    AZURE_DOCUMENTINTELLIGENCE_API_KEY = os.getenv("AZURE_DOCUMENTINTELLIGENCE_API_KEY")

    return DocumentIntelligenceClient(
        endpoint=AZURE_DOCUMENTINTELLIGENCE_ENDPOINT,
        credential=AzureKeyCredential(AZURE_DOCUMENTINTELLIGENCE_API_KEY),
    )


def _setup_semantic_kernel() -> sk.Kernel:
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_MULTIMODAL_MODEL_DEPLOYMENT_NAME = os.getenv(
        "AZURE_OPENAI_MULTIMODAL_MODEL_DEPLOYMENT_NAME"
    )
    AZURE_OPENAI_TEXT_EMBEDDING_MODEL_DEPLOYMENT_NAME = os.getenv(
        "AZURE_OPENAI_TEXT_EMBEDDING_MODEL_DEPLOYMENT_NAME"
    )

    kernel = sk.Kernel()

    kernel.add_service(
        service=AzureTextEmbedding(
            service_id="openai__embedder",
            deployment_name=AZURE_OPENAI_TEXT_EMBEDDING_MODEL_DEPLOYMENT_NAME,
            endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
        )
    )

    kernel.add_service(
        service=AzureChatCompletion(
            service_id="openai__chatter",
            deployment_name=AZURE_OPENAI_MULTIMODAL_MODEL_DEPLOYMENT_NAME,
            endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
        )
    )

    print(f"Kernel services: {kernel.services}")

    return kernel


def _setup_memory_store() -> MemoryStoreBase:
    chroma_client = HttpClient(host="localhost", port="8000")
    _ = chroma_client.get_or_create_collection("lease_document")
    return ChromaMemoryStore(client_settings=chroma_client.get_settings())


def _parse_args(raw_args: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filepath",
        required=True,
        help="Relative path to the original file out of which data needs to be extracted and analyzed",
    )
    parser.add_argument(
        "-e",
        "--from-prior-experiment-run",
        required=False,
        help="Expects the id of a prior experiment run from which to re-use extracted data",
    )

    return parser.parse_args(raw_args)


def _read_extracted_data(filepath: str) -> dict:
    with open(filepath) as file:
        file_content = file.read()

        data = json.loads(file_content)

    return data


def _print_extracted_data(data: dict):
    for idx, style in enumerate(data["styles"]):
        print(
            "Document contains {} content".format(
                "handwritten" if style["isHandwritten"] else "no handwritten"
            )
        )

    for page in data["pages"]:
        for line_idx, line in enumerate(page["lines"]):
            print(
                "...Line # {} has text content '{}'".format(
                    line_idx, line["content"].encode("utf-8")
                )
            )

    for table_idx, table in enumerate(data["tables"]):
        print(
            "Table # {} has {} rows and {} columns".format(
                table_idx, table["rowCount"], table["columnCount"]
            )
        )

        for cell in table["cells"]:
            print(
                "...Cell[{}][{}] has content '{}'".format(
                    cell["rowIndex"],
                    cell["columnIndex"],
                    cell["content"].encode("utf-8"),
                )
            )


def extract_data_from_file(
    document_intelligence_client: DocumentIntelligenceClient,
    filepath: str,
    output_destination: str,
    from_prior_experiment_run: str,
) -> str:
    output_file = None

    if from_prior_experiment_run:
        print(
            f"Reusing extracted data of experiment run {from_prior_experiment_run}..."
        )
        prior_experiment_run_output_file = f"{get_experiment_run_destination(EXPERIMENT_NAME, from_prior_experiment_run)}/results.json"
        result = _read_extracted_data(prior_experiment_run_output_file)
        _print_extracted_data(result)
    else:
        with open(filepath, "rb") as file:
            file_content = file.read()

        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout",
            AnalyzeDocumentRequest(bytes_source=file_content),
        )

        result = poller.result().as_dict()
        _print_extracted_data(result)

    if EXPERIMENT_SAVE_RESULTS:
        output_file = f"{output_destination}/results.json"
        with open(output_file, "w") as f:
            json.dump(result, f)

    return output_file


async def embed_extracted_data(filepath: str):
    if not EXPERIMENT_SAVE_RESULTS:
        print(f"Data has to be saved before it can be embedded!")
    else:
        kernel = _setup_semantic_kernel()
        memory_store = _setup_memory_store()
        memory = SemanticTextMemory(
            storage=memory_store,
            embeddings_generator=kernel.get_service(service_id="openai__embedder"),
        )

        data = _read_extracted_data(filepath)

        for i, chunk in enumerate(data["paragraphs"]):
            if chunk["spans"][0]["length"] > 50:
                await memory.save_information(
                    collection="lease_document", id=f"chunk {i}", text=chunk["content"]
                )

        return memory


async def get_query_response(
    memory: SemanticTextMemory, collection: str, query: str, limit: int
):
    results = await memory.search(collection=collection, query=query, limit=limit)

    for result in results:
        print(f"Text: {result.text} \nRelevance: {result.relevance}")


def handle_cli():
    load_dotenv()
    args = _parse_args(sys.argv[1:])
    filepath = args.filepath
    from_prior_experiment_run = args.from_prior_experiment_run

    experiment_run, output_destination = make_experiment_run(
        EXPERIMENT_NAME, EXPERIMENT_SAVE_RESULTS
    )

    document_intelligence_client = _setup_document_intelligence_client()
    extracted_data_file = extract_data_from_file(
        document_intelligence_client,
        filepath,
        output_destination,
        from_prior_experiment_run,
    )

    memory = asyncio.run(embed_extracted_data(extracted_data_file))

    asyncio.run(
        get_query_response(
            memory,
            "lease_document",
            "What is the name of the premises/unit that the tenant is leasing?",
            5,
        )
    )

    end_experiment_run(EXPERIMENT_NAME, experiment_run)
