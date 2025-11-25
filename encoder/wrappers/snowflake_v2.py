from unittest.mock import Base
from encoder.wrappers.base import BaseWrapper

from torch import no_grad, nn
from numpy import ndarray


class SnowFlakeArcticV2Wrapper(BaseWrapper):
    
    def __init__(self, model_name):
        super().__init__(model_name)
        
    def tokenize(self, text):
        return self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        
    def embed(self, t):
        with no_grad():
            model_output = self.model(**t)[0][:, 0]
        
        # print(model_output)
        embeddings = nn.functional.normalize(model_output, p=2, dim=1).cpu().numpy()
        # print(embeddings)
        return embeddings