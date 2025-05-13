import asyncio
from typing import Optional

REASONING_MODELS = [
    "o1",
    "o1-2024-12-17",
    "o3-mini-2025-01-31",
    "o3-mini",
    "o4-mini",
    "o4-mini-2025-04-16",
    "o3",
    "o3-2025-04-16"
]

PERPLEXITY_MODELS = [
    "sonar-pro",
    "sonar-reasoning-pro",
    "r1-1776"
]

LLM_MODEL_COSTS = {
    # o1 models
    "o1-mini": {
        "input": 3 / 10 ** 6,
        "cached_input": 1.5 / 10 ** 6,
        "output": 12 / 10 ** 6,
    },
    "o1": {
        "input": 15 / 10 ** 6,
        "cached_input": 7.5 / 10 ** 6,
        "output": 60 / 10 ** 6,
    },
    "o1-2024-12-17": {
        "input": 15 / 10 ** 6,
        "cached_input": 7.5 / 10 ** 6,
        "output": 60 / 10 ** 6,
    },
    # o3 models
    "o3": {
        "input": 10.0 / 10 ** 6,
        "cached_input": 2.5 / 10 ** 6,
        "output": 40.0 / 10 ** 6,
    },
    "o3-2025-04-16": {
        "input": 10.0 / 10 ** 6,
        "cached_input": 2.5 / 10 ** 6,
        "output": 40.0 / 10 ** 6,
    },
    # o4 mini models
    "o4-mini": {
        "input": 1.1 / 10 ** 6,
        "cached_input": 0.275 / 10 ** 6,
        "output": 4.4 / 10 ** 6,
    },
    "o4-mini-2025-04-16": {
        "input": 1.1 / 10 ** 6,
        "cached_input": 0.275 / 10 ** 6,
        "output": 4.4 / 10 ** 6,
    },
    # o3 mini models
    "o3-mini": {
        "input": 1.1 / 10 ** 6,
        "cached_input": 0.55 / 10 ** 6,
        "output": 4.4 / 10 ** 6,
    },
    "o3-mini-2025-01-31": {
        "input": 1.1 / 10 ** 6,
        "cached_input": 0.55 / 10 ** 6,
        "output": 4.4 / 10 ** 6,
    },
    # gpt-4.1 models
    "gpt-4.1-mini": {
        "input": 0.40 / 10 ** 6,
        "cached_input": 0.10 / 10 ** 6,
        "output": 1.60 / 10 ** 6,
    },
    "gpt-4.1": {
        "input": 2.00 / 10 ** 6,
        "cached_input": 0.50 / 10 ** 6,
        "output": 8.00 / 10 ** 6,
    },
    "gpt-4.1-nano": {
        "input": 0.10 / 10 ** 6,
        "cached_input": 0.025 / 10 ** 6,
        "output": 0.40 / 10 ** 6,
    },
    # gpt-4o models
    "gpt-4o": {
        "input": 2.5 / 10 ** 6,
        "cached_input": 1.25 / 10 ** 6,
        "output": 10 / 10 ** 6,
    },
    "gpt-4o-2024-11-20": {
        "input": 2.5 / 10 ** 6,
        "cached_input": 1.25 / 10 ** 6,
        "output": 10 / 10 ** 6,
    },
    "gpt-4o-2024-08-06": {
        "input": 2.5 / 10 ** 6,
        "cached_input": 1.25 / 10 ** 6,
        "output": 10 / 10 ** 6,
    },
    "gpt-4o-mini": {
        "input": 0.15 / 10 ** 6,
        "cached_input": 0.075 / 10 ** 6,
        "output": 0.6 / 10 ** 6,
    },
    # embedding models
    "text-embedding-3-small": {
        "input": 0.02 / 10 ** 6,
    },
    "text-embedding-3-large": {
        "input": 0.13 / 10 ** 6,
    },
    "text-embedding-ada-002": {
        "input": 0.1 / 10 ** 6,
    }
}

class ModelUsageAsync:
    def __init__(self, model_costs=LLM_MODEL_COSTS, web_search_costs=None):
        """
        Args:
            model_costs (dict[str, dict[str, float]]): Cost per token for each model (make sure to scale to 1 token)
            web_search_costs (dict[str, dict[str, float]]): cost per web search for each model (make sure to scale to 1 token)
        """
        self.token_usage = {}
        self.web_search_usage = {}

        self.model_costs = model_costs
        self.web_search_costs = web_search_costs
        
        self.logs = []
        self.lock = asyncio.Lock()

    async def add_tokens(
        self, 
        model: str, 
        input_tokens: int = 0, 
        output_tokens: int = 0, 
        cached_tokens: int = 0, 
        reasoning_tokens: int = 0, 
        label: Optional[str] = None
    ) -> None:
        """
        Add the number of tokens used to the token usage dictionary.
        
        Args:
            model (str): The model used.
            input_tokens (int, optional): The number of input tokens used. Defaults to 0.
            output_tokens (int, optional): The number of output tokens used. Defaults to 0.
            cached_tokens (int, optional): The number of cached tokens used. Defaults to 0.
            reasoning_tokens (int, optional): The number of reasoning tokens used. Defaults to 0.
            label (str, optional): The label of the tokens used. Defaults to None.
        """
        async with self.lock:
            if model not in self.token_usage:
                self.token_usage[model] = {
                    "input": 0, 
                    "output": 0, 
                    "cached": 0, 
                    "reasoning": 0
                }
            
            self.token_usage[model]["input"] += input_tokens
            self.token_usage[model]["output"] += output_tokens
            self.token_usage[model]["cached"] += cached_tokens
            self.token_usage[model]["reasoning"] += reasoning_tokens
            
            if label is not None:
                self.logs.append({
                    "model": model, 
                    "input_tokens": input_tokens, 
                    "output_tokens": output_tokens, 
                    "cached_tokens": cached_tokens, 
                    "reasoning_tokens": reasoning_tokens,
                    "label": label
                })

    async def add_web_search_usage(
        self,
        model: str,
        search_context_size: str
    ) -> None:
        """
        Add to the tally of web search usage if the web_search_costs are provided.

        Args:
            model (str): The model used.
            search_context_size (str): The size of the search context.
        """
        if self.web_search_costs is not None:
            if model not in self.web_search_usage:
                self.web_search_usage[model] = {}
            if search_context_size not in self.web_search_usage[model]:
                self.web_search_usage[model][search_context_size] = 0
            
            # add one to the tally of number of searches for a given model and context size
            self.web_search_usage[model][search_context_size] += 1
        
    async def get_cost(self) -> float:
        """
        Get the total cost of the tokens used. Note, that embedding models don't have "output" costs, so we use .get() to return 0 for those.
        
        Cached tokens are half price of the input token's price.
        
        Reasoning tokens are the same price as the output token's price.
        
        Returns:
            float: Total cost of the tokens used.
        """
        async with self.lock:
            # Check if all models are supported
            for model in self.token_usage:
                if model not in self.model_costs:
                    raise ValueError(f"Model {model} is not supported.")
            
            # Reasoning tokens are already counted in the output tokens
            # Get the total cost for input/output of each model
            cost_of_input_output = sum(
                [
                    (self.token_usage[model]["input"] - self.token_usage[model]["cached"]) * self.model_costs[model].get("input", 0) + # Subtract cached tokens from input
                    self.token_usage[model]["output"] * self.model_costs[model].get("output", 0) +
                    self.token_usage[model]["cached"] * self.model_costs[model].get("cached_input", 0)
                    for model in self.token_usage 
                ]
            ) 

            # Get the additional cost of web searches
            cost_of_web_searches = 0
            if self.web_search_costs is not None:
                for model in self.web_search_usage:
                    for search_context_size in self.web_search_usage[model]:
                        # multiply the number of searches of each model and context size by the cost of the search
                        cost_of_web_searches += self.web_search_usage[model][search_context_size] * self.web_search_costs[model].get(search_context_size, 0)

            # Return the total cost - the cost of input/output + the cost of web searches
            return cost_of_input_output + cost_of_web_searches

    async def get_tokens_used(self) -> int:
        """
        Get the total number of tokens used.
        
        Returns:
            int: Total number of tokens used.
        """
        async with self.lock:
            return sum(
                [
                    self.token_usage[model]["input"] + # Input tokens
                    self.token_usage[model]["output"] # Output tokens
                    for model in self.token_usage
                ]
            )

    async def get_web_searches_performed(self) -> int:
        """
        Get the total number of web searches performed.
        
        Returns:
            int: Total number of web searches performed.
        """
        async with self.lock:
            total_number_of_searches = 0
            for model in self.web_search_usage:
                for search_context_size in self.web_search_usage[model]:
                    total_number_of_searches += self.web_search_usage[model][search_context_size]
            return total_number_of_searches
