import asyncio
import unittest

from client import async_stream, collect_async_response, collect_sync_response, simulate_llm_response, sync_stream
from client import transform_sync_stream, transform_async_stream
from client import safe_sync_stream, safe_async_stream
from client import LLMStreamingClient


class TestSyncStream(unittest.TestCase):
    def test_sync_stream_yields_tokens(self):
        prompt = "Hello, world!"
        tokens = list(sync_stream(prompt))
        self.assertGreater(len(tokens), 0, "sync_stream should yield at least one token.")
        self.assertEqual(tokens, ["Hello,", "world!"])

class TestAsyncStream(unittest.IsolatedAsyncioTestCase):
    async def test_async_stream_yields_tokens(self):
        prompt = "Hello async world!"
        tokens = []
        async for token in async_stream(prompt):
            tokens.append(token)
        self.assertGreater(len(tokens), 0, "async_stream should yield at least one token.")
        self.assertEqual(tokens, ["Hello", "async", "world!"])
        
class TestLLMResponse(unittest.IsolatedAsyncioTestCase):
    async def test_simulate_llm_response_yields_expected_tokens(self):
        prompt = "Tell me a joke"
        # For testing purposes, we expect the simulated LLM to yield these tokens:
        expected_tokens = ["Why", "did", "the", "chicken", "cross", "the", "road", "?"]
        tokens = []
        async for token in simulate_llm_response(prompt):
            tokens.append(token)
        self.assertEqual(tokens, expected_tokens, "simulate_llm_response should yield the expected tokens.")
        
        
class TestCollectResponse(unittest.TestCase):
    def test_collect_sync_response(self):
        prompt = "This is a test"
        response = collect_sync_response(prompt)
        # The sync_stream splits the prompt into tokens and joining them with a space
        self.assertEqual(response, "This is a test")

class TestCollectAsyncResponse(unittest.IsolatedAsyncioTestCase):
    async def test_collect_async_response(self):
        prompt = "Async test"
        response = await collect_async_response(prompt)
        # Similarly, async_stream yields tokens that when joined, reconstruct the prompt
        self.assertEqual(response, "Async test")
        
class TestTransformStream(unittest.TestCase):
    def test_transform_sync_stream(self):
        prompt = "hello world"
        # Transformation function: convert each token to uppercase.
        def to_upper(token):
            return token.upper()
        transformed_tokens = list(transform_sync_stream(sync_stream(prompt), to_upper))
        self.assertEqual(transformed_tokens, ["HELLO", "WORLD"],
                         "Synchronous stream tokens should be transformed to uppercase.")

class TestTransformAsyncStream(unittest.IsolatedAsyncioTestCase):
    async def test_transform_async_stream(self):
        prompt = "hello async"
        def to_upper(token):
            return token.upper()
        transformed_tokens = []
        async for token in transform_async_stream(async_stream(prompt), to_upper):
            transformed_tokens.append(token)
        self.assertEqual(transformed_tokens, ["HELLO", "ASYNC"],
                         "Asynchronous stream tokens should be transformed to uppercase.")

# Define a synchronous generator that fails after yielding some tokens.
def failing_sync_stream():
    yield "Sync1"
    yield "Sync2"
    raise Exception("Sync failure")

# Define an asynchronous generator that fails after yielding some tokens.
async def failing_async_stream():
    yield "Async1"
    yield "Async2"
    raise Exception("Async failure")

class TestSafeStream(unittest.TestCase):
    def test_safe_sync_stream_handles_error(self):
        # on_error callback for sync streams
        on_error = lambda e: f"[ERROR: {str(e)}]"
        tokens = list(safe_sync_stream(failing_sync_stream(), on_error))
        # We expect the tokens to be yielded until the error, then the error token.
        self.assertEqual(tokens, ["Sync1", "Sync2", "[ERROR: Sync failure]"])

class TestSafeAsyncStream(unittest.IsolatedAsyncioTestCase):
    async def test_safe_async_stream_handles_error(self):
        # on_error callback for async streams
        on_error = lambda e: f"[ERROR: {str(e)}]"
        tokens = []
        async for token in safe_async_stream(failing_async_stream(), on_error):
            tokens.append(token)
        self.assertEqual(tokens, ["Async1", "Async2", "[ERROR: Async failure]"])
        
        
class TestLLMStreamingClientSync(unittest.TestCase):
    def test_stream_sync(self):
        client = LLMStreamingClient()
        prompt = "Hello world"
        tokens = list(client.stream_sync(prompt))
        # Our sync_stream splits the prompt by spaces.
        self.assertEqual(tokens, prompt.split(), "stream_sync should yield tokens split by spaces.")

    def test_complete_sync(self):
        client = LLMStreamingClient()
        prompt = "Hello world"
        response = client.complete_sync(prompt)
        self.assertEqual(response, prompt, "complete_sync should reassemble tokens into the original prompt.")

class TestLLMStreamingClientAsync(unittest.IsolatedAsyncioTestCase):
    async def test_astream(self):
        client = LLMStreamingClient()
        prompt = "Hello async world"
        tokens = []
        async for token in client.astream(prompt):
            tokens.append(token)
        self.assertEqual(tokens, prompt.split(), "astream should yield tokens split by spaces.")

    async def test_acomplete(self):
        client = LLMStreamingClient()
        prompt = "Hello async world"
        response = await client.acomplete(prompt)
        self.assertEqual(response, prompt, "acomplete should reassemble tokens into the original prompt.")

class TestLLMStreamingClientCallback(unittest.TestCase):
    def test_stream_sync_with_callback(self):
        client = LLMStreamingClient()
        prompt = "Hello callback"
        callback_results = []

        def callback(token):
            callback_results.append(token)

        # Call stream_sync with an optional callback parameter
        tokens = list(client.stream_sync(prompt, callback=callback))
        # Verify tokens are yielded as expected
        self.assertEqual(tokens, prompt.split(),
                         "stream_sync should yield tokens split by spaces.")
        # Verify that the callback was invoked for each token
        self.assertEqual(callback_results, prompt.split(),
                         "Callback should be invoked for each token.")

class TestLLMStreamingClientAsyncCallback(unittest.IsolatedAsyncioTestCase):
    async def test_astream_with_callback(self):
        client = LLMStreamingClient()
        prompt = "Async callback test"
        callback_results = []

        def callback(token):
            callback_results.append(token)

        tokens = []
        # Call astream with an optional callback parameter
        async for token in client.astream(prompt, callback=callback):
            tokens.append(token)
        # Verify tokens are yielded as expected
        self.assertEqual(tokens, prompt.split(),
                         "astream should yield tokens split by spaces.")
        # Verify that the callback was invoked for each token
        self.assertEqual(callback_results, prompt.split(),
                         "Callback should be invoked for each token.")

        
class TestLLMStreamingClientWithAsyncCallback(unittest.IsolatedAsyncioTestCase):
    async def test_astream_with_async_callback(self):
        client = LLMStreamingClient()
        prompt = "Async callback test"
        callback_results = []

        async def async_callback(token):
            # Simulate some async processing
            await asyncio.sleep(0.01)
            callback_results.append(token)

        tokens = []
        async for token in client.astream(prompt, callback=async_callback):
            tokens.append(token)
        # Verify tokens are yielded as expected
        self.assertEqual(tokens, prompt.split(),
                         "astream should yield tokens split by spaces.")
        # Verify that the async callback was invoked for each token
        self.assertEqual(callback_results, prompt.split(),
                         "Async callback should be invoked for each token.")


if __name__ == "__main__":
    unittest.main()
