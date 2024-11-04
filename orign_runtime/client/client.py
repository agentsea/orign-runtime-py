from typing import Optional, List, AsyncIterator
import websockets
import json

from orign_runtime.client.config import Config
from orign.models import (
    SamplingParams,
    ChatResponse,
    Prompt,
    TokenResponse,
    ChatRequest,
    MessageItem,
)


class Stream:
    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        kind: Optional[str] = None,
        orign_addr: str = Config.ORIGN_ADDR,
    ):
        self.model = model
        self.provider = provider
        self.kind = kind
        self.orign_addr = orign_addr
        self.websocket = None

        if not self.kind and not self.model:
            raise ValueError("Either 'kind' or 'model' must be provided")

        # Construct the WebSocket URL
        self.ws_url = f"ws://{self.orign_addr}/v1/chat/stream/{self.model or self.kind}"
        if self.provider:
            self.ws_url += f"?provider={self.provider}"

    async def connect(self):
        """Establish WebSocket connection if not already connected"""
        if not self.websocket:
            self.websocket = await websockets.connect(self.ws_url)

    async def close(self):
        """Close the WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

    async def chat(
        self,
        msg: Optional[str] = None,
        prompt: Optional[Prompt] = None,
        batch: Optional[List[Prompt]] = None,
        sampling_params: Optional[SamplingParams] = None,
        stream_tokens: bool = False,
    ) -> AsyncIterator[ChatResponse | TokenResponse]:
        await self.connect()

        # Create ChatRequest object
        request = ChatRequest(
            sampling_params=sampling_params or SamplingParams(),
            stream=stream_tokens,
            prompt=prompt,
            batch=batch,
        )

        # If msg is provided, convert it to a Prompt
        if msg:
            request.prompt = Prompt(messages=[MessageItem(role="user", content=msg)])
        elif prompt:
            request.prompt = prompt
        elif batch:
            request.batch = batch

        # Send the request
        await self.websocket.send(request.model_dump_json())

        # Yield responses as they arrive
        try:
            while True:
                response = await self.websocket.recv()
                response_data = json.loads(response)

                if "token" in response_data:
                    yield TokenResponse(**response_data)
                else:
                    yield ChatResponse(**response_data)

                # Check if this is the final message
                if response_data.get("done", False):
                    break

        except websockets.exceptions.ConnectionClosed:
            await self.close()
            raise ConnectionError("WebSocket connection closed unexpectedly")
