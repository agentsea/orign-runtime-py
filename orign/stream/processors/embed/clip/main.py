from typing import AsyncGenerator

from pydantic_settings import BaseSettings

from orign.stream.processors.base_aio import EmbeddingModel, EmbeddingRequest, EmbeddingResponses


class ClipConfig(BaseSettings):
    pass


class Clip(EmbeddingModel[ClipConfig]):
    """
    Clip backend for embedding processing.
    """

    def load(self, config: ClipConfig):
        self.config = config


    async def process(self, msg: EmbeddingRequest) -> AsyncGenerator[EmbeddingResponses, None]:
        print("Processing message", flush=True)


if __name__ == "__main__":
    import asyncio

    processor = Clip()
    config = ClipConfig()
    asyncio.run(processor.run(config))
