import os

import asyncio
import semantic_kernel as sk

from dotenv import load_dotenv, find_dotenv
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextCompletion

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

kernel = sk.Kernel()

kernel.add_service(
    service=AzureChatCompletion(
        service_id="azure_openai__gpt4__chat_completion",
        deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY
    )
)

kernel.add_service(
    service=AzureTextCompletion(
        service_id="azure_openai__gpt4__text_completion",
        deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY
    )
)

print(kernel.services)

async def get_prompt_result() -> None:
    prompt = """
    {{$input}} is the capital of 
    """

    target_service_id = "azure_openai__gpt4__text_completion"

    execution_config = kernel.get_service(target_service_id).instantiate_prompt_execution_settings(
            service_id=target_service_id,
            max_tokens=100,
            temperature=0,
            seed=42
        )

    generate_capital_city_text = kernel.add_function(
        prompt=prompt,
        plugin_name="Generate_City_Completion",
        function_name="generate_city_completion",
        execution_settings=execution_config
    )

    response = await kernel.invoke(generate_capital_city_text, input="Paris")

    if response:
        print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(get_prompt_result())
