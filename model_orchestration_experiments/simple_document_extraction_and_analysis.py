import os
import sys
import json

import hashlib

import asyncio
import argparse

from dotenv import load_dotenv

from chromadb import Client, HttpClient
from chromadb.config import Settings

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

from semantic_kernel import Kernel as SemanticKernel
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextCompletion,
    AzureTextEmbedding,
)
from semantic_kernel.connectors.memory.chroma import ChromaMemoryStore
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin
from semantic_kernel.functions import KernelArguments
from semantic_kernel.memory.memory_store_base import MemoryStoreBase
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.prompt_template.input_variable import InputVariable

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


def _setup_semantic_kernel() -> SemanticKernel:
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_MULTIMODAL_MODEL_DEPLOYMENT_NAME = os.getenv(
        "AZURE_OPENAI_MULTIMODAL_MODEL_DEPLOYMENT_NAME"
    )
    AZURE_OPENAI_TEXT_EMBEDDING_MODEL_DEPLOYMENT_NAME = os.getenv(
        "AZURE_OPENAI_TEXT_EMBEDDING_MODEL_DEPLOYMENT_NAME"
    )

    kernel = SemanticKernel()

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

    # print(f"Kernel services: {kernel.services}")

    return kernel


def _setup_memory_store(filepath: str) -> tuple[MemoryStoreBase, str]:
    chroma_client = HttpClient(host="localhost", port="8000")
    collection_name = hashlib.sha1(filepath.encode("utf-8")).hexdigest()
    _ = chroma_client.get_or_create_collection(collection_name)
    return (
        ChromaMemoryStore(client_settings=chroma_client.get_settings()),
        collection_name,
    )


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


def _print_chat_action(action: str, text: str):
    DIVIDER = f"\n{'-' * 50}\n"
    print(f"{DIVIDER}{action}:{DIVIDER}")
    print(f"{text}")


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
        # _print_extracted_data(result)
    else:
        with open(filepath, "rb") as file:
            file_content = file.read()

        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout",
            AnalyzeDocumentRequest(bytes_source=file_content),
        )

        result = poller.result().as_dict()
        # _print_extracted_data(result)

    if EXPERIMENT_SAVE_RESULTS:
        output_file = f"{output_destination}/results.json"
        with open(output_file, "w") as f:
            json.dump(result, f)

    return output_file


async def add_memory(
    kernel: SemanticKernel, filepath: str
) -> tuple[SemanticTextMemory, str]:
    if not EXPERIMENT_SAVE_RESULTS:
        print(f"Data has to be saved before it can be embedded!")

        return None, None
    else:
        memory_store, collection_name = _setup_memory_store(filepath)
        memory = SemanticTextMemory(
            storage=memory_store,
            embeddings_generator=kernel.get_service(service_id="openai__embedder"),
        )

        kernel.add_plugin(TextMemoryPlugin(memory), "TextMemoryPlugin")

        data = _read_extracted_data(filepath)

        for i, chunk in enumerate(data["paragraphs"]):
            if chunk["spans"][0]["length"] > 50:
                await memory.save_information(
                    collection=collection_name, id=f"chunk {i}", text=chunk["content"]
                )

        return memory, collection_name


async def chat(kernel: SemanticKernel, collection: str, query: str):
    prompt_with_context_plugin = """
        Use the following pieces of context to answer the users question.
        This is the only information that you should use to answer the question, do not reference information outside of this context.
        If the information required to answer the question is not provided in the context, just say that "I don't know", don't try to make up an answer.
        ----------------
        Context: {{recall $question}}
        ----------------
        User question: {{$question}}
        ----------------
        Answer:
    """

    target_service_id = "openai__chatter"

    execution_config = kernel.get_service(
        target_service_id
    ).instantiate_prompt_execution_settings(
        service_id=target_service_id, max_tokens=500, temperature=0, seed=42
    )

    prompt_template_config = PromptTemplateConfig(
        template=prompt_with_context_plugin,
        name="chat",
        template_format="semantic-kernel",
        input_variables=[
            InputVariable(
                name="question", description="The user input", is_required=True
            ),
            InputVariable(
                name="context", description="The conversation history", is_required=True
            ),
        ],
        execution_settings=execution_config,
    )

    chatbot_with_context_plugin = kernel.add_function(
        prompt_template_config=prompt_template_config,
        plugin_name="chatPluginWithContextPlugin",
        function_name="chatbot_with_context_plugin",
        execution_settings=execution_config,
    )

    context = KernelArguments(
        question=query, collection=collection, relevance=0.2, limit=5
    )
    answer = await kernel.invoke(chatbot_with_context_plugin, context)

    _print_chat_action("Question", query)
    _print_chat_action("Answer", answer)


def handle_cli():
    load_dotenv()
    args = _parse_args(sys.argv[1:])
    filepath = args.filepath
    from_prior_experiment_run = args.from_prior_experiment_run

    experiment_run, output_destination = make_experiment_run(
        EXPERIMENT_NAME, EXPERIMENT_SAVE_RESULTS
    )

    kernel = _setup_semantic_kernel()

    document_intelligence_client = _setup_document_intelligence_client()
    extracted_data_file = extract_data_from_file(
        document_intelligence_client,
        filepath,
        output_destination,
        from_prior_experiment_run,
    )

    _, collection = asyncio.run(add_memory(kernel, extracted_data_file))

    # examples for a Lease Agreement document
    asyncio.run(
        chat(
            kernel,
            collection,
            "What is the name of the tenant that entered the lease agreement?",
        )
    )
    asyncio.run(
        chat(
            kernel,
            collection,
            "What is the name of the space that the tenant is leasing?",
        )
    )
    asyncio.run(
        chat(kernel, collection, "What is the start date of the lease agreement?")
    )

    end_experiment_run(EXPERIMENT_NAME, experiment_run)
