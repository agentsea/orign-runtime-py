from typing import AsyncGenerator
from io import BytesIO
import base64
from pydantic_settings import BaseSettings
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests

from orign_runtime.stream.processors.base_aio import EmbeddingModel, EmbeddingRequest, EmbeddingResponses
from orign.models import Embedding, EmbeddingResponse

class SentenceTFConfig(BaseSettings):
    model: str = "clip-ViT-B-32"
    device: str = "cuda"


class SentenceTF(EmbeddingModel[SentenceTFConfig]):
    """
    SentenceTF backend for embedding processing.
    """

    def load(self, config: SentenceTFConfig):
        self.config = config
        print(f"Loading SentenceTF model {config.model} on device {config.device}")
        self.model = SentenceTransformer(config.model)


    async def process(self, msg: EmbeddingRequest) -> AsyncGenerator[EmbeddingResponses, None]:
        print("Processing message", flush=True)

        embeddings = []
        if msg.image:
            if msg.image.startswith("http://") or msg.image.startswith("https://"):
                # It's a URL, download with requests
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                'AppleWebKit/537.36 (KHTML, like Gecko) '
                                'Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(msg.image, headers=headers)
                response.raise_for_status()
                image_data = response.content
                img = Image.open(BytesIO(image_data))
            elif msg.image.startswith("data:"):
                header, data = msg.image.split(',', 1)
                image_data = base64.b64decode(data)
                img = Image.open(BytesIO(image_data))
            else:
                raise ValueError(f"Invalid image format: {msg.image}")

            embeddings.append(self.model.encode(img))

        if msg.text:
            embeddings.append(self.model.encode(msg.text))
        

        if len(embeddings) == 0:
            raise ValueError("No embeddings found")
        
        emb_instances = []
        for embedding in embeddings:
            emb_instance = Embedding(
                object="embedding",
                index=0,
                embedding=embedding.tolist()
            )
            emb_instances.append(emb_instance)
        
        # Create the EmbeddingResponse instance
        response = EmbeddingResponse(
            object="list",
            data=emb_instances,
            model=msg.model if msg.model else "default-model",
            request_id=msg.request_id
        )
        yield response
    

if __name__ == "__main__":
    import asyncio

    processor = SentenceTF()
    config = SentenceTFConfig()
    asyncio.run(processor.run(config))
