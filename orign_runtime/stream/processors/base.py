# base.py
from abc import ABC, abstractmethod
from typing import TypeVar, List, Type, Generic
import traceback

from pydantic import BaseModel

from orign.models import ErrorResponse
from ..queue.factory import get_message_consumer, get_message_producer
S = TypeVar("S", bound=BaseModel)


class ModelBackend(ABC, Generic[S]):
    def __init__(self):
        self.config = None
        self.engine = None
        self.producer = None
        self.consumer = None

    @abstractmethod
    def initialize_engine(self):
        """Initialize the language model engine."""
        pass

    @abstractmethod
    def process_message(self, id: str, msg: S):
        """Process a single message from the consumer."""
        pass

    @abstractmethod
    def accepts(self) -> S:
        """The schema accepted by the backend."""
        pass

    @abstractmethod
    def produces(self) -> List[Type[BaseModel]]:
        """The schema(s) produced by the backend."""
        pass

    def main(self):
        """Main loop for processing messages."""
        print("Starting main()")
        self.initialize_engine()
        print("Initialized Engine")
        
        self.consumer = get_message_consumer(self.config)
        self.producer = get_message_producer(self.config)

        schema = self.supported_schema()

        try:
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    print(f"Consumer error: {msg.error()}")
                    continue

                base_request_id = f"{msg.topic()}-{msg.partition()}-{msg.offset()}"
                try:
                    message = schema.model_validate_json(msg.value())
                except Exception as e:
                    print(f"Validation error: {e}")
                    
                    self.producer.produce(ErrorResponse(error=(f"Validation error: {e}"), request_id=base_request_id))
                    self.consumer.commit(message=msg)
                    continue
                
                try:
                    self.process_message(id=base_request_id, msg=message)
                except Exception as e:
                    error_trace = traceback.format_exc()
                    print(f"Error processing message: {e} -- {error_trace}")
                    self.producer.produce(ErrorResponse(error=str(e), request_id=base_request_id, traceback=error_trace))

                self.consumer.commit(message=msg)
                print(f"Committed message {base_request_id}")

        except KeyboardInterrupt:
            pass

        finally:
            print("Closing consumer and producer")
            self.consumer.close()
            self.producer.flush()