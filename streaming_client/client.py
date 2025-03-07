import asyncio
import random

def sync_stream(prompt: str):
    """
    A basic synchronous streaming function that yields tokens.
    """
    tokens = prompt.split() if prompt else ["No", "prompt", "provided"]
    for token in tokens:
        yield token

async def async_stream(prompt: str):
    """
    A basic asynchronous streaming function that yields tokens.
    Mimics asynchronous behavior with a small delay between tokens.
    """
    tokens = prompt.split() if prompt else ["No", "prompt", "provided"]
    for token in tokens:
        # Simulate asynchronous processing delay
        await asyncio.sleep(random.uniform(0.5, 1))
        yield token


async def simulate_llm_response(prompt: str):
    """
    Simulates an LLM's token-by-token streaming response.
    Regardless of the prompt, for demonstration, it yields a fixed set of tokens.
    """
    # This simulated response is static. In a real LLM call, these tokens would be generated dynamically.
    tokens = ["Why", "did", "the", "chicken", "cross", "the", "road", "?"]
    for token in tokens:
        await asyncio.sleep(random.uniform(0.05, 0.10))  # Simulate a slight delay between token generations
        yield token
        
def collect_sync_response(prompt: str) -> str:
    """
    Collects tokens from the synchronous stream and joins them into a full response string.
    """
    tokens = list(sync_stream(prompt))
    # Joining tokens with a space (adjust as needed based on tokenization)
    return " ".join(tokens)

async def collect_async_response(prompt: str) -> str:
    """
    Collects tokens from the asynchronous stream and joins them into a full response string.
    """
    tokens = []
    async for token in async_stream(prompt):
        tokens.append(token)
    return " ".join(tokens)


def transform_sync_stream(generator, transform_func):
    """
    Wraps a synchronous generator to apply a transformation function on each token.
    """
    for token in generator:
        yield transform_func(token)

async def transform_async_stream(async_generator, transform_func):
    """
    Wraps an asynchronous generator to apply a transformation function on each token.
    """
    async for token in async_generator:
        yield transform_func(token)
        
        
def safe_sync_stream(generator, on_error=None):
    """
    Wraps a synchronous generator to handle errors gracefully.
    If an error occurs, yields the result of on_error(e) if provided.
    """
    try:
        for token in generator:
            yield token
    except Exception as e:
        if on_error:
            yield on_error(e)
        else:
            raise

async def safe_async_stream(async_gen, on_error=None):
    """
    Wraps an asynchronous generator to handle errors gracefully.
    If an error occurs, yields the result of on_error(e) if provided.
    """
    try:
        async for token in async_gen:
            yield token
    except Exception as e:
        if on_error:
            yield on_error(e)
        else:
            raise
        
        
class LLMStreamingClient:
    """
    A unified client that provides both synchronous and asynchronous streaming responses,
    as well as complete (aggregated) responses for an LLM.
    """
    def stream_sync(self, prompt: str):
        """
        Synchronously yields tokens from the LLM.
        """
        return sync_stream(prompt)

    def complete_sync(self, prompt: str) -> str:
        """
        Synchronously collects all tokens into a complete response.
        """
        return collect_sync_response(prompt)

    async def astream(self, prompt: str):
        """
        Asynchronously yields tokens from the LLM.
        """
        async for token in async_stream(prompt):
            yield token

    async def acomplete(self, prompt: str) -> str:
        """
        Asynchronously collects all tokens into a complete response.
        """
        return await collect_async_response(prompt)