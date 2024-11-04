from io import BytesIO
import base64
import numpy as np
import requests
from PIL import Image
import traceback
import time
from typing import AsyncGenerator

import easyocr
from pydantic_settings import BaseSettings

from orign.models import OCRRequest, OCRResponse, BoundingBox, ErrorResponse
from orign_runtime.stream.processors.base_aio import OCRModel, OCRResponses


class EasyOCRConfig(BaseSettings):
    device: str = "cuda"
    gpu: bool = True
    lang_list: list[str] = ["en"]
    quantize: bool = False


class EasyOCR(OCRModel[EasyOCRConfig]):
    """
    EasyOCR backend for OCR processing.
    """

    def load(self, config: EasyOCRConfig):
        params = {
            "lang_list": config.lang_list,
            "gpu": config.gpu,
            "quantize": config.quantize,
        }
        if config.device == "cuda":
            params["gpu"] = True

        # def __init__(self, lang_list, gpu=True, model_storage_directory=None,
        #             user_network_directory=None, detect_network="craft", 
        #             recog_network='standard', download_enabled=True, 
        #             detector=True, recognizer=True, verbose=True, 
        #             quantize=True, cudnn_benchmark=False):
        self.reader = easyocr.Reader(**params)
        print("EasyOCR engine initialized", flush=True)


    async def process(self, msg: OCRRequest) -> AsyncGenerator[OCRResponses, None]:
        print("Processing message", flush=True)
        try:
            start_time = time.time()

            # Handle image input
            if msg.image.startswith("http://") or msg.image.startswith("https://"):
                # It's a URL, download with requests
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                
                response = requests.get(msg.image, headers=headers)
                response.raise_for_status()
                image_data = response.content
                input_image = np.array(Image.open(BytesIO(image_data)))
            else:
                # Try to decode as base64; if it fails, assume it's a file path
                try:
                    image_data = base64.b64decode(msg.image)
                    input_image = np.array(Image.open(BytesIO(image_data)))
                except base64.binascii.Error:
                    # Not base64, assume it's a file path
                    input_image = msg.image

            # def readtext(self, image, decoder = 'greedy', beamWidth= 5, batch_size = 1,\
            #              workers = 0, allowlist = None, blocklist = None, detail = 1,\
            #              rotation_info = None, paragraph = False, min_size = 20,\
            #              contrast_ths = 0.1,adjust_contrast = 0.5, filter_ths = 0.003,\
            #              text_threshold = 0.7, low_text = 0.4, link_threshold = 0.4,\
            #              canvas_size = 2560, mag_ratio = 1.,\
            #              slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
            #              width_ths = 0.5, y_ths = 0.5, x_ths = 1.0, add_margin = 0.1, 
            #              threshold = 0.2, bbox_min_score = 0.2, bbox_min_size = 3, max_candidates = 0,
            #              output_format='standard'):

            # Run the OCR model
            results = self.reader.readtext(
                input_image,
                detail=1 if msg.detail else 0,
                paragraph=msg.paragraph,
                contrast_ths=0.1,
                adjust_contrast=0.5,
                text_threshold=0.7,
                low_text=0.4,
                link_threshold=0.4,
            )

            # Filter results based on min_confidence
            min_confidence = msg.min_confidence if msg.min_confidence is not None else 0.0

            # Map results to data model
            if msg.detail:
                # results is a list of tuples: (bbox, text, confidence)
                bounding_boxes = [
                    BoundingBox(
                        points=[list(map(int, point)) for point in box],
                        text=text,
                        confidence=float(confidence)
                    ) for box, text, confidence in results
                    if float(confidence) >= min_confidence
                ]
                ocr_response = OCRResponse(
                    type="OCRResponse",
                    request_id=msg.request_id,
                    results=bounding_boxes,
                    processing_time=time.time() - start_time,
                    error=None
                )
            else:
                # results is a list of strings
                texts = [
                    text for text, confidence in results
                    if float(confidence) >= min_confidence
                ]
                ocr_response = OCRResponse(
                    type="OCRResponse",
                    request_id=msg.request_id,
                    results=texts,
                    processing_time=time.time() - start_time,
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

    backend = EasyOCR()
    config = EasyOCRConfig()
    asyncio.run(backend.run(config))
