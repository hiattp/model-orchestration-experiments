import os
import sys
import json

import semantic_kernel as sk
import argparse

from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

from .utils import (
    make_experiment_run,
    end_experiment_run,
    get_experiment_run_destination,
)

EXPERIMENT_NAME = "SIMPLE_DOCUMENT_EXTRACTION_AND_ANALYSIS"
EXPERIMENT_SAVE_RESULTS = True


def _setup_client() -> DocumentIntelligenceClient:
    AZURE_DOCUMENTINTELLIGENCE_ENDPOINT = os.getenv(
        "AZURE_DOCUMENTINTELLIGENCE_ENDPOINT"
    )
    AZURE_DOCUMENTINTELLIGENCE_API_KEY = os.getenv("AZURE_DOCUMENTINTELLIGENCE_API_KEY")

    return DocumentIntelligenceClient(
        endpoint=AZURE_DOCUMENTINTELLIGENCE_ENDPOINT,
        credential=AzureKeyCredential(AZURE_DOCUMENTINTELLIGENCE_API_KEY),
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
):
    if from_prior_experiment_run:
        print(
            f"Reusing extracted data of experiment run {from_prior_experiment_run}..."
        )
        prior_experiment_run_output_file = f"{get_experiment_run_destination(EXPERIMENT_NAME, from_prior_experiment_run)}/results.json"
        with open(prior_experiment_run_output_file) as file:
            file_content = file.read()

            result = json.loads(file_content)
            # _print_extracted_data(result)
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


def handle_cli():
    load_dotenv()
    args = _parse_args(sys.argv[1:])
    filepath = args.filepath
    from_prior_experiment_run = args.from_prior_experiment_run

    experiment_run, output_destination = make_experiment_run(
        EXPERIMENT_NAME, EXPERIMENT_SAVE_RESULTS
    )

    document_intelligence_client = _setup_client()
    extract_data_from_file(
        document_intelligence_client,
        filepath,
        output_destination,
        from_prior_experiment_run,
    )

    end_experiment_run(EXPERIMENT_NAME, experiment_run)
