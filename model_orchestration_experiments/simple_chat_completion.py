import os
import sys

import asyncio
import semantic_kernel as sk
import argparse

from dotenv import load_dotenv

from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextCompletion,
)

from .utils import make_experiment_run, end_experiment_run

EXPERIMENT_NAME = "SIMPLE_CHAT_COMPLETION"
EXPERIMENT_SAVE_RESULTS = False


def _setup_kernel() -> sk.Kernel:
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_MULTIMODAL_MODEL_DEPLOYMENT_NAME = os.getenv(
        "AZURE_OPENAI_MULTIMODAL_MODEL_DEPLOYMENT_NAME"
    )

    kernel = sk.Kernel()

    kernel.add_service(
        service=AzureChatCompletion(
            service_id="azure_openai__gpt4__chat_completion",
            deployment_name=AZURE_OPENAI_MULTIMODAL_MODEL_DEPLOYMENT_NAME,
            endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
        )
    )

    kernel.add_service(
        service=AzureTextCompletion(
            service_id="azure_openai__gpt4__text_completion",
            deployment_name=AZURE_OPENAI_MULTIMODAL_MODEL_DEPLOYMENT_NAME,
            endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
        )
    )

    print(f"Kernel services: {kernel.services}")

    return kernel


def _parse_args(raw_args: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Simple input to the following chat prompt: <<input>> is the capital of",
    )

    return parser.parse_args(raw_args)


async def get_prompt_result(kernel: sk.Kernel, input: str) -> None:
    prompt = """
    {{$input}} is the capital of 
    """

    target_service_id = "azure_openai__gpt4__text_completion"

    execution_config = kernel.get_service(
        target_service_id
    ).instantiate_prompt_execution_settings(
        service_id=target_service_id, max_tokens=100, temperature=0, seed=42
    )

    generate_capital_city_text = kernel.add_function(
        prompt=prompt,
        plugin_name="Generate_City_Completion",
        function_name="generate_city_completion",
        execution_settings=execution_config,
    )

    response = await kernel.invoke(generate_capital_city_text, input="Paris")

    if response:
        print(f"Response: {response}")


def handle_cli():
    load_dotenv()
    args = _parse_args(sys.argv[1:])
    input = args.input

    experiment_run, output_destination = make_experiment_run(
        EXPERIMENT_NAME, EXPERIMENT_SAVE_RESULTS
    )

    kernel = _setup_kernel()
    asyncio.run(get_prompt_result(kernel, input))

    end_experiment_run(EXPERIMENT_NAME, experiment_run)
