from typing import List, Union, Optional, Dict, Any

from pydantic import BaseModel

# === Chat Request ===

class ImageUrlContent(BaseModel):
    """Image URL content for chat requests"""
    url: str

class ContentItem(BaseModel):
    """Content item for chat requests"""
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrlContent] = None

class MessageItem(BaseModel):
    """Message item for chat requests"""
    role: str
    content: Union[str, List[ContentItem]]  # Updated to allow a list of ContentItem

class Prompt(BaseModel):
    """Prompt for chat requests"""
    messages: List[MessageItem]

class SamplingParams(BaseModel):
    """Sampling parameters for chat requests"""

    n: int = 1
    best_of: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None
    min_tokens: int = 0
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    detokenize: bool = True
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: Optional[int] = None

class Usage(BaseModel):
    """Usage for chat requests"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatRequest(BaseModel):
    """Chat request"""

    type: str = "ChatRequest"
    request_id: Optional[str] = None
    model: Optional[str] = None
    kind: Optional[str] = None
    backend: Optional[str] = None
    namespace: Optional[str] = None
    prompt: Optional[Prompt] = None
    batch: Optional[List[Prompt]] = None
    max_tokens: int = 512
    sampling_params: SamplingParams = SamplingParams()
    stream: bool = False
    user_id: Optional[str] = None
    output_topic: Optional[str] = None
    output_partition: Optional[int] = None
    

class Choice(BaseModel):
    """Individual choice in the token response"""

    index: int
    text: str
    tokens: Optional[List[str]] = None
    token_ids: Optional[List[int]] = None
    logprobs: Optional[List[Dict[Union[int, str], Any]]] = None
    finish_reason: Optional[str] = None

class ChatResponse(BaseModel):
    """Chat response"""

    type: str = "ChatResponse"
    request_id: str
    choices: List[Choice]
    trip_time: Optional[float] = None
    usage: Optional[Usage] = None

class TokenResponse(BaseModel):
    """Token response"""

    type: str = "TokenResponse"
    request_id: str
    tokens: List[str]
    token_ids: Optional[List[int]] = None
    logprobs: Optional[List[Dict[Union[int, str], Any]]] = None
    usage: Optional[Usage] = None

# === OCR Request ===

class OCRRequest(BaseModel):
    """Simple OCR request following EasyOCR patterns"""

    image: str
    languages: List[str]  # e.g. ['en'], ['ch_sim', 'en']
    gpu: bool = True
    detail: bool = True  # True returns bounding boxes, False returns just text
    paragraph: bool = False  # Merge text into paragraphs
    min_confidence: Optional[float] = 0.0

class BoundingBox(BaseModel):
    """Coordinates for text location: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]"""

    points: List[List[int]]  # List of 4 points (8 coordinates total)
    text: str
    confidence: float

class OCRResponse(BaseModel):
    """Response containing detected text and locations"""

    results: Union[List[BoundingBox], List[str]]  # List[str] if detail=False
    processing_time: Optional[float]
    error: Optional[str]


# === Errors ===

class ErrorResponse(BaseModel):
    """Error response"""

    type: str = "ErrorResponse"
    request_id: str
    error: str
    traceback: Optional[str] = None