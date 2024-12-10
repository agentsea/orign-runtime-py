import requests
from PIL import Image
from io import BytesIO
import base64

def delivery_report(err, msg):
    """Callback function for producer to report message delivery."""
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")


async def open_image_from_input_async(input_str: str) -> Image.Image:
    """
    Opens an image from a given URL or base64-encoded string.

    Parameters:
    input_str (str): The URL of the image or base64 string.

    Returns:
    Image.Image: The opened PIL Image object.
    """
    print(f"\n\n!!!Opening image from input: {input_str}\n\n", flush=True)
    if input_str.startswith("data:image/"):
        # It's a base64 images
        header, encoded_data = input_str.split(",", 1)
        image_data = base64.b64decode(encoded_data)
        return Image.open(BytesIO(image_data)).convert("RGB")
    else:
        # It's a URL
        response = requests.get(input_str)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")


def open_image_from_input(input_str: str) -> Image.Image:
    """
    Opens an image from a given URL or base64-encoded string.

    Parameters:
    input_str (str): The URL of the image or base64 string.

    Returns:
    Image.Image: The opened PIL Image object.
    """
    if input_str.startswith("data:image/"):
        # It's a base64 image
        header, encoded_data = input_str.split(",", 1)
        image_data = base64.b64decode(encoded_data)
        return Image.open(BytesIO(image_data)).convert("RGB")
    else:
        # It's a URL
        response = requests.get(input_str)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
