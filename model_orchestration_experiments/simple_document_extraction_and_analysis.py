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
from semantic_kernel.connectors.ai.google.google_ai import (
    GoogleAIChatCompletion,
    GoogleAITextEmbedding,
)
from semantic_kernel.connectors.ai.google.vertex_ai import (
    VertexAIChatCompletion,
    VertexAITextEmbedding,
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
    bboxes_overlap,
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


def _setup_semantic_kernel(service: str) -> SemanticKernel:
    kernel = SemanticKernel()

    if service == "openai":
        AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        AZURE_OPENAI_CHAT_COMPLETION_MODEL_DEPLOYMENT_NAME = os.getenv(
            "AZURE_OPENAI_CHAT_COMPLETION_MODEL_DEPLOYMENT_NAME"
        )
        AZURE_OPENAI_TEXT_EMBEDDING_MODEL_DEPLOYMENT_NAME = os.getenv(
            "AZURE_OPENAI_TEXT_EMBEDDING_MODEL_DEPLOYMENT_NAME"
        )
        kernel.add_service(
            service=AzureTextEmbedding(
                service_id="embedder",
                deployment_name=AZURE_OPENAI_TEXT_EMBEDDING_MODEL_DEPLOYMENT_NAME,
                endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
            )
        )
        kernel.add_service(
            service=AzureChatCompletion(
                service_id="chatter",
                deployment_name=AZURE_OPENAI_CHAT_COMPLETION_MODEL_DEPLOYMENT_NAME,
                endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
            )
        )
    elif service == "googleai":
        GOOGLE_AI_GEMINI_API_KEY = os.getenv("GOOGLE_AI_GEMINI_API_KEY")
        GOOGLE_AI_GEMINI_CHAT_COMPLETION_MODEL_DEPLOYMENT_NAME = os.getenv(
            "GOOGLE_AI_GEMINI_CHAT_COMPLETION_MODEL_DEPLOYMENT_NAME"
        )
        GOOGLE_AI_GEMINI_TEXT_EMBEDDING_MODEL_DEPLOYMENT_NAME = os.getenv(
            "GOOGLE_AI_GEMINI_TEXT_EMBEDDING_MODEL_DEPLOYMENT_NAME"
        )

        kernel.add_service(
            service=GoogleAITextEmbedding(
                embedding_model_id=GOOGLE_AI_GEMINI_TEXT_EMBEDDING_MODEL_DEPLOYMENT_NAME,
                api_key=GOOGLE_AI_GEMINI_API_KEY,
                service_id="embedder",
            )
        )
        kernel.add_service(
            service=GoogleAIChatCompletion(
                gemini_model_id=GOOGLE_AI_GEMINI_CHAT_COMPLETION_MODEL_DEPLOYMENT_NAME,
                api_key=GOOGLE_AI_GEMINI_API_KEY,
                service_id="chatter",
            )
        )
    elif service == "vertexai":
        VERTEX_AI_PROJECT_ID = os.getenv("VERTEX_AI_PROJECT_ID")
        VERTEX_AI_REGION = os.getenv("VERTEX_AI_REGION")
        VERTEX_AI_GEMINI_CHAT_COMPLETION_MODEL_DEPLOYMENT_NAME = os.getenv(
            "VERTEX_AI_GEMINI_CHAT_COMPLETION_MODEL_DEPLOYMENT_NAME"
        )
        VERTEX_AI_GEMINI_TEXT_EMBEDDING_MODEL_DEPLOYMENT_NAME = os.getenv(
            "VERTEX_AI_GEMINI_TEXT_EMBEDDING_MODEL_DEPLOYMENT_NAME"
        )

        kernel.add_service(
            service=VertexAITextEmbedding(
                project_id=VERTEX_AI_PROJECT_ID,
                region=VERTEX_AI_REGION,
                embedding_model_id=VERTEX_AI_GEMINI_TEXT_EMBEDDING_MODEL_DEPLOYMENT_NAME,
                service_id="embedder",
            )
        )
        kernel.add_service(
            service=VertexAIChatCompletion(
                project_id=VERTEX_AI_PROJECT_ID,
                region=VERTEX_AI_REGION,
                gemini_model_id=VERTEX_AI_GEMINI_CHAT_COMPLETION_MODEL_DEPLOYMENT_NAME,
                service_id="chatter",
            )
        )
    else:
        raise RuntimeError("No model family to use!")

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
        "-s",
        "--service",
        required=True,
        choices=["openai", "googleai", "vertexai"],
        help="The AI Service to use",
    )
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
    kernel: SemanticKernel, output_destination: str, filepath: str
) -> tuple[SemanticTextMemory, str]:
    if not EXPERIMENT_SAVE_RESULTS:
        print(f"Data has to be saved before it can be embedded!")
        return None, None
    else:
        memory_store, collection_name = _setup_memory_store(filepath)
        memory = SemanticTextMemory(
            storage=memory_store,
            embeddings_generator=kernel.get_service(service_id="embedder"),
        )
        kernel.add_plugin(TextMemoryPlugin(memory), "TextMemoryPlugin")
        data = _read_extracted_data(filepath)

        # Build a map of tables by page and their polygons (all polygons from all bounding regions)
        tables_by_page = {}
        table_polygons_by_page = {}
        for table in data.get("tables", []):
            for region in table.get("boundingRegions", []):
                page = region["pageNumber"]
                polygon = region.get("polygon")
                if polygon:
                    table_polygons_by_page.setdefault(page, []).append(polygon)
            # For saving, still group tables by the first region's page
            page = (
                table["boundingRegions"][0]["pageNumber"]
                if table.get("boundingRegions")
                else 1
            )
            tables_by_page.setdefault(page, []).append(table)

        # Group paragraphs by page, skipping those that overlap with any table polygon on the same page
        paragraphs_by_page = {}
        for paragraph in data.get("paragraphs", []):
            overlaps = False
            for region in paragraph.get("boundingRegions", []):
                page = region["pageNumber"]
                paragraph_polygon = region.get("polygon")
                if paragraph_polygon:
                    for table_polygon in table_polygons_by_page.get(page, []):
                        if bboxes_overlap(paragraph_polygon, table_polygon):
                            overlaps = True
                            break
                if overlaps:
                    break
            if not overlaps and paragraph.get("boundingRegions"):
                # Use the first region's page for grouping
                page = paragraph["boundingRegions"][0]["pageNumber"]
                paragraphs_by_page.setdefault(page, []).append(paragraph["content"])

        # Prepare to save chunks to a JSON file
        chunks = []

        # Save paragraph chunks (grouped, max 2000 chars, do not split original paragraphs)
        for page, paragraphs in paragraphs_by_page.items():
            if paragraphs:
                chunk = ""
                chunk_idx = 0
                for paragraph in paragraphs:
                    if len(chunk) + len(paragraph) + 1 > 2000 and chunk:
                        chunk_id = f"page_{page}_paragraphs_{chunk_idx}"
                        await memory.save_information(
                            collection=collection_name,
                            id=chunk_id,
                            text=chunk.strip(),
                        )
                        chunks.append(
                            {
                                "id": chunk_id,
                                "type": "paragraphs",
                                "page": page,
                                "text": chunk.strip(),
                            }
                        )
                        chunk = ""
                        chunk_idx += 1
                    if chunk:
                        chunk += "\n"
                    chunk += paragraph
                if chunk:
                    chunk_id = f"page_{page}_paragraphs_{chunk_idx}"
                    await memory.save_information(
                        collection=collection_name,
                        id=chunk_id,
                        text=chunk.strip(),
                    )
                    chunks.append(
                        {
                            "id": chunk_id,
                            "type": "paragraphs",
                            "page": page,
                            "text": chunk.strip(),
                        }
                    )

        # Save tables (each table as a chunk)
        for page, tables in tables_by_page.items():
            for j, table in enumerate(tables):
                rows = [[] for _ in range(table["rowCount"])]
                for cell in table["cells"]:
                    row_idx = cell["rowIndex"]
                    col_idx = cell["columnIndex"]
                    while len(rows[row_idx]) <= col_idx:
                        rows[row_idx].append("")
                    rows[row_idx][col_idx] = cell["content"].replace("\n", " ")
                table_str = "\n".join("|".join(row) for row in rows)
                chunk_id = f"page_{page}_table_{j}"
                await memory.save_information(
                    collection=collection_name,
                    id=chunk_id,
                    text=table_str,
                )
                chunks.append(
                    {"id": chunk_id, "type": "table", "page": page, "text": table_str}
                )

        chunks_file = f"{output_destination}/chunks.json"
        with open(chunks_file, "w") as f:
            json.dump(chunks, f)

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

    target_service_id = "chatter"

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
    service = args.service
    filepath = args.filepath
    from_prior_experiment_run = args.from_prior_experiment_run

    experiment_run, output_destination = make_experiment_run(
        EXPERIMENT_NAME, EXPERIMENT_SAVE_RESULTS
    )

    kernel = _setup_semantic_kernel(service)

    document_intelligence_client = _setup_document_intelligence_client()
    extracted_data_file = extract_data_from_file(
        document_intelligence_client,
        filepath,
        output_destination,
        from_prior_experiment_run,
    )

    _, collection = asyncio.run(
        add_memory(kernel, output_destination, extracted_data_file)
    )

    asyncio.run(
        chat(
            kernel,
            collection,
            """
                Summarize in a table the key details of the lease agreement. The columns of the table that you must include are listed below, together with some instructions and constraints:
                - Tenant Name: this represents the name of the tenant entering the lease agreement
                - Building Address: ensure it is a valid address
                - Rentable Area: this represents the total area in square feet/total square footage of the premises being rented and you must ensure it is a number
                - Unit: this represents the name of the premises or the term through which the premises is referred to
                - Lease Start Date: you must ensure it is a date in the format YYYY-MM-DD
                - Lease End Date: you must ensure it is a date in the format YYYY-MM-DD
                - Term in Months: ensure it is a number
            """,
        )
    )

    asyncio.run(
        chat(
            kernel,
            collection,
            """
                Summarize in a table the charging schedule of the lease agreement. The fields of the table that you must include are listed below, together with some of their constraints that you must respect:
                - From Date: ensure it is a date in the format YYYY-MM-DD
                - To Date: ensure it is a date in the format YYYY-MM-DD
                - Monthly Base Rent: ensure it is a number
                - Base Rent Currency: ensure it is a valid currency
            """,
        )
    )

    asyncio.run(
        chat(
            kernel,
            collection,
            "Summarize in a table the key details of the termination or break options the tenant has. The fields of the table must be: Date of Notice, Date of Termination, Rent Penalty (USD), Trigger Reason.",
        )
    )

    end_experiment_run(EXPERIMENT_NAME, experiment_run)
