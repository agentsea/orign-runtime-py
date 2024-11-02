from typing import AsyncGenerator

from pydantic_settings import BaseSettings

from orign.stream.processors.base_aio import EmbeddingModel, EmbeddingRequest, EmbeddingResponses


class SentenceTFConfig(BaseSettings):
    pass


class SentenceTF(EmbeddingModel[SentenceTFConfig]):
    """
    SentenceTF backend for embedding processing.
    """

    def load(self, config: SentenceTFConfig):
        self.config = config


    async def process(self, msg: EmbeddingRequest) -> AsyncGenerator[EmbeddingResponses, None]:
        print("Processing message", flush=True)


if __name__ == "__main__":
    import asyncio

    processor = SentenceTF()
    config = SentenceTFConfig()
    asyncio.run(processor.run(config))
