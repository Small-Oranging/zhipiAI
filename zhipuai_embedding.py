from __future__ import annotations
import logging
from typing import Dict, List, Any
from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator

logger = logging.getLogger(__name__)

class ZhipuAIEmbeddings(BaseModel, Embeddings):

    client: Any

    def embed_query(self, text: str) -> List[float]:
        embeddings = self.client.embeddings.create(model="embedding-2",input=text)
        return embeddings.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]
    
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        from zhipuai import ZhipuAI
        values["client"] = ZhipuAI()
        return values    
