import base64
import requests
import traceback
import time
from typing import AsyncGenerator

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from pydantic_settings import BaseSettings

from orign.models import OCRRequest, OCRResponse, BoundingBox, ErrorResponse
from orign_runtime.stream.processors.base_aio import OCRModel, OCRResponses


class DoctrConfig(BaseSettings):
    det_arch: str = "fast_base"
    reco_arch: str = "crnn_vgg16_bn"
    pretrained: bool = True


class Doctr(OCRModel[DoctrConfig]):
    """
    Doctr backend for OCR processing.
    """

    def load(self, config: DoctrConfig):
        self.config = config
        self.model = model = ocr_predictor(**config.model_dump())
        print("EasyOCR engine initialized", flush=True)


    async def process(self, msg: OCRRequest) -> AsyncGenerator[OCRResponses, None]:
        print("Processing message", flush=True)
        try:
            start_time = time.time()

            # Process the image
            if msg.image.startswith('http'):
                # It's a URL
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(msg.image, headers=headers)
                image_data = response.content
            else:
                # Assume it's a base64-encoded image
                image_data = base64.b64decode(msg.image)

            # Read the image using DocumentFile
            imgs = DocumentFile.from_images(image_data)
            # Analyze the image
            result = self.model(imgs)

            # Prepare the results
            if msg.detail:
                # Return bounding boxes and text
                results = []
                for page in result.pages:
                    height, width = page.dimensions
                    for block in page.blocks:
                        for line in block.lines:
                            for word in line.words:
                                # Get the word's geometry (normalized coordinates)
                                ((x_min, y_min), (x_max, y_max)) = word.geometry
                                # Convert normalized coordinates to absolute pixel values
                                x1 = int(x_min * width)
                                y1 = int(y_min * height)
                                x2 = int(x_max * width)
                                y2 = y1
                                x3 = x2
                                y3 = int(y_max * height)
                                x4 = x1
                                y4 = y3
                                points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                                # Filter by minimum confidence if specified
                                if msg.min_confidence is None or word.confidence >= msg.min_confidence:
                                    results.append(BoundingBox(
                                        points=points,
                                        text=word.value,
                                        confidence=word.confidence,
                                    ))
            else:
                # Return just text
                results = []
                for page in result.pages:
                    page_text = page.export()['content']
                    results.append(page_text)

            processing_time = time.time() - start_time

            ocr_response = OCRResponse(
                type="OCRResponse",
                request_id=msg.request_id,
                results=results,
                processing_time=processing_time,
                error=None
            )

            # Send the response
            yield ocr_response

        except Exception as e:
            error_trace = traceback.format_exc()
            error_response = ErrorResponse(
                type="ErrorResponse",
                request_id=msg.request_id,
                error=str(e),
                traceback=error_trace,
            )
            yield error_response

if __name__ == "__main__":
    import asyncio

    backend = Doctr()
    config = DoctrConfig()
    asyncio.run(backend.run(config))
