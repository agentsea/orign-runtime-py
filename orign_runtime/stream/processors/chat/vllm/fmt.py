from abc import ABC, abstractmethod
from typing import NamedTuple, Optional, List

from .qwen_vision import process_vision_info, fetch_image, extract_vision_info
from orign.models import ChatRequest, Prompt
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image

from orign_runtime.stream.util import open_image_from_input_async


class ModelRequestData(NamedTuple):
    prompt: str
    stop_token_ids: Optional[List[str]]
    image_data: List[Image.Image]
    chat_template: Optional[str]


class MessageFormatter(ABC):
    
    @abstractmethod
    async def format(self, request: ChatRequest) -> ModelRequestData:
        pass


class Qwen2VLMessageFormatter(MessageFormatter):
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    async def format(self, prompt: Prompt) -> ModelRequestData:
        print(f"Formatting prompt: {prompt}")
        # Convert Prompt model to Qwen's expected format
        formatted_messages = []
        image_urls = []
        
        for message in prompt.messages:
            if isinstance(message.content, str):
                formatted_messages.append({
                    "role": message.role,
                    "content": message.content
                })
            else:  # List[ContentItem]
                formatted_content = []
                for item in message.content:
                    if item.type == "text":
                        formatted_content.append({
                            "type": "text",
                            "text": item.text
                        })
                    elif item.type == "image_url" and item.image_url:
                        image_urls.append(item.image_url.url)
                        formatted_content.append({
                            "type": "image",
                            "image": item.image_url.url
                        })
                
                formatted_messages.append({
                    "role": message.role,
                    "content": formatted_content
                })

        print("!type of formatted_messages: ", type(formatted_messages), flush=True)
        print("!formatted_messages: ", formatted_messages, flush=True)

        prompt = self.processor.apply_chat_template(formatted_messages,
                                            tokenize=False,
                                            add_generation_prompt=True)

        print("!type of prompt: ", type(prompt), flush=True)
        print("!prompt: ", prompt, flush=True)

        stop_token_ids = None

        extracted_vision_info = extract_vision_info(formatted_messages)
        print("!type of extracted_vision_info: ", type(extracted_vision_info), flush=True)
        print("!extracted_vision_info: ", extracted_vision_info, flush=True)

        for vision_info in extracted_vision_info:
            print("!type of vision_info: ", type(vision_info), flush=True)
            print("!vision_info: ", vision_info, flush=True)
            print("fetching image vision info...")
            image_data = fetch_image(vision_info)
            print("!type of image_data: ", type(image_data), flush=True)
            print("!image_data: ", image_data, flush=True)

        if process_vision_info is None:
            image_data = [await open_image_from_input_async(url) for url in image_urls]
        else:
            print("image_urls: ", image_urls, flush=True)
            print(f"Processing vision info for {len(image_urls)} images", flush=True)
            image_data, _ = process_vision_info(formatted_messages)
            print("past process_vision_info")
            print(f"Processed vision info for {len(image_urls)} images: {image_data}", flush=True)

        print("!type of image_data: ", type(image_data), flush=True)

        return ModelRequestData(
            prompt=prompt,
            stop_token_ids=stop_token_ids,
            image_data=image_data,
            chat_template=None,
        )


class MolmoMessageFormatter(MessageFormatter):
    async def format(self, prompt: Prompt) -> ModelRequestData:
        prompt_text = ""
        images = []

        # Handle messages within the prompt
        for msg_entry in prompt.messages:
            if isinstance(msg_entry.content, str):
                prompt_text += msg_entry.content + "\n"
            elif isinstance(msg_entry.content, list):
                for content_item in msg_entry.content:
                    if content_item.type == "text" and content_item.text:
                        prompt_text += content_item.text + "\n"
                    elif (
                        content_item.type == "image_url" and content_item.image_url
                    ):
                        try:
                            image_url = content_item.image_url.url
                            image = await open_image_from_input_async(image_url)
                            print(f"Downloaded image from URL '{image_url}'")
                            images.append(image)
                        except Exception as e:
                            print(f"Failed to load image: {e}")
                            continue
                    else:
                        print(f"Unknown content item type: {content_item.type}")
            elif isinstance(msg_entry.content, ContentItem):
                content_item = msg_entry.content
                if content_item.type == "text" and content_item.text:
                    prompt_text += content_item.text + "\n"
                elif content_item.type == "image_url" and content_item.image_url:
                    try:
                        image_url = content_item.image_url.url
                        image = await open_image_from_input_async(image_url)
                        print(f"Downloaded image from URL '{image_url}'")
                        images.append(image)
                    except Exception as e:
                        print(f"Failed to load image: {e}")
                        continue
                elif content_item.type == "image_base64" and content_item.data:
                    try:
                        base64_data = content_item.data
                        image = await open_image_from_input_async(base64_data)
                        print("Decoded base64 image")
                        images.append(image)
                    except Exception as e:
                        print(f"Failed to load image: {e}")
                        continue
                else:
                    print(f"Unknown content item type: {content_item.type}")
            else:
                print(
                    f"Unexpected content type in message: {type(msg_entry.content)}"
                )

        if not prompt_text.strip():
            raise ValueError("No valid content found in message item")
            
        stop_token_ids = None

        return ModelRequestData(
            prompt=prompt_text,
            stop_token_ids=stop_token_ids,
            image_data=images,
            chat_template=None,
        )


MODEL_FORMATTER_MAP = {
    "qwen2_vl": Qwen2VLMessageFormatter,
    "molmo": MolmoMessageFormatter,
}

MODEL_TYPE_MAP = {
    # Qwen2-VL
    "Qwen/Qwen2-VL-2B": "qwen2_vl",
    "Qwen/Qwen2-VL-2B-Instruct": "qwen2_vl",
    "Qwen/Qwen2-VL-7B": "qwen2_vl",
    "Qwen/Qwen2-VL-7B-Instruct": "qwen2_vl",
    "Qwen/Qwen2-VL-72B": "qwen2_vl",
    "Qwen/Qwen2-VL-72B-Instruct": "qwen2_vl",
    "Qwen/Qwen2-VL-2B-Instruct-AWQ": "qwen2_vl",
    "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4": "qwen2_vl",
    "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8": "qwen2_vl",
    "Qwen/Qwen2-VL-7B-Instruct-AWQ": "qwen2_vl",
    "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4": "qwen2_vl",
    "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8": "qwen2_vl",
    "Qwen/Qwen2-VL-72B-Instruct-AWQ": "qwen2_vl",
    "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4": "qwen2_vl",
    "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8": "qwen2_vl",
    # Molmo
    "allenai/Molmo-72B-0924": "molmo",
    "allenai/Molmo-7B-D-0924": "molmo",
    "allenai/Molmo-7B-O-0924": "molmo",
    "allenai/MolmoE-1B-0924": "molmo",
}

