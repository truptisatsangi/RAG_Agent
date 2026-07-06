from retrieval_service import RetrievalService
from prompt_service import PromptService
from llm_service import LLMService

class AskService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        prompt_service: PromptService,
        llm_service: LLMService,
    ):
        self.retrieval_service = retrieval_service
        self.prompt_service = prompt_service
        self.llm_service = llm_service