from typing import Optional, List, Iterator
from orign.server.models import SamplingParams, ChatResponse, Prompt, TokenResponse
from .config import Config

class Stream:
    def __init__(self, 
                 model: Optional[str] = None, 
                 sampling_params: Optional[SamplingParams] = None, 
                 kind: Optional[str] = None,
                 orign_addr: str = Config.ORIGN_ADDR):

        self.model = model
        self.sampling_params = sampling_params
        self.kind = kind
        self.orign_addr = orign_addr

        if not self.kind and not self.model:
            raise ValueError("Either 'kind' or 'model' must be provided")
    
    def chat(self, msg: Optional[str] = None,
             prompt: Optional[Prompt] = None, 
             batch: Optional[List[Prompt]] = None) -> Iterator[ChatResponse | TokenResponse]:
        pass
