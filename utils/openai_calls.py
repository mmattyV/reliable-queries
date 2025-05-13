import asyncio
from openai import AsyncOpenAI, ChatCompletion
from typing import Optional
import traceback

from .model_costs import ModelUsageAsync, REASONING_MODELS, PERPLEXITY_MODELS

async def call_openai_structured(
    openai_client: AsyncOpenAI,
    model: str,
    llm_usage: Optional[ModelUsageAsync] = None,
    llm_usage_label: Optional[str] = None,
    max_retries: int = 3,
    base_delay: int = 5,
    completion_timeout: int = 90,
    web_search_bool: bool = False,
    web_search_context_size: str = "high",
    **kwargs
) -> Optional[ChatCompletion]:
    """
    Call OpenAI's API to get a structured response.

    Args:
        openai_client (AsyncOpenAI): The OpenAI client to use.
        model (str): The model to use.
        llm_usage (Optional[ModelUsageAsync], optional): The ModelUsageAsync object to use. Defaults to None.
        llm_usage_label (Optional[str], optional): The label to use for the LLM usage. Defaults to None.
        max_retries (int, optional): The maximum number of retries. Defaults to 3.
        base_delay (int, optional): The base delay between retries. Defaults to 5.
        completion_timeout (int, optional): The timeout for the API call. Defaults to 90.
        web_search_bool (bool, optional): Whether to use web search. Defaults to False.
        web_search_context_size (str, optional): The context size for web search. Defaults to "high".
        **kwargs: Additional keyword arguments to pass to the API call.
        
    Returns:
        Optional[ChatCompletion]: The response from the API call.
    """
    if model in REASONING_MODELS:
        # Reasoning models don't support temperature
        if "temperature" in kwargs:
            del kwargs["temperature"]
    else:
        # Only supported by reasoning models
        if "reasoning_effort" in kwargs:
            del kwargs["reasoning_effort"]
    
    for _ in range(max_retries):
        try:
            # Wrap the API call in asyncio.wait_for to enforce the completion timeout.
            completion = await asyncio.wait_for(
                openai_client.beta.chat.completions.parse(
                    model=model,
                    **kwargs,
                ),
                timeout=completion_timeout
            )
            
            if llm_usage is not None:
                if model not in PERPLEXITY_MODELS:
                    await llm_usage.add_tokens(
                        model=model,
                        input_tokens=completion.usage.prompt_tokens,
                        output_tokens=completion.usage.completion_tokens,
                        cached_tokens=completion.usage.prompt_tokens_details.cached_tokens,
                        reasoning_tokens=completion.usage.completion_tokens_details.reasoning_tokens,
                        label=llm_usage_label
                    )
                else:
                    await llm_usage.add_tokens(
                        model=model,
                        input_tokens=completion.usage.prompt_tokens,
                        output_tokens=completion.usage.completion_tokens,
                        cached_tokens=0,
                        reasoning_tokens=0,
                        label=llm_usage_label
                    )
                
                if web_search_bool:
                    await llm_usage.add_web_search_usage(
                        model=model,
                        search_context_size=web_search_context_size
                    )
            
            return completion
        except asyncio.TimeoutError:
            delay = base_delay
            if llm_usage_label is not None:
                print(f"API call ({llm_usage_label}) timed out after {completion_timeout} seconds. Retrying in {delay} seconds...")
            else:
                print(f"API call timed out after {completion_timeout} seconds. Retrying in {delay} seconds...")
            await asyncio.sleep(delay)
        except Exception as e:
            delay = base_delay
            if llm_usage_label is not None:
                print(f"API call for model {model} ({llm_usage_label}) failed: {e}. Retrying in {delay} seconds...")
            else:
                print(f"API call for model {model} failed: {e}. Retrying in {delay} seconds...")
                print("Full stack trace:")
                print(traceback.format_exc())
            await asyncio.sleep(delay)

    print("Max retries exceeded")

    return None
