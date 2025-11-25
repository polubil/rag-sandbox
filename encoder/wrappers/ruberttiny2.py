from unittest.mock import Base
from torch import no_grad, nn
from numpy import ndarray
from encoder.wrappers.base import BaseWrapper


class RuBERTTiny2Wrapper(BaseWrapper):
    
    def __init__(self, model_name):
        super().__init__(model_name)
        
    def tokenize(self, text):
        return self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        
    def embed(self, t):
        with no_grad():
            model_output = self.model(**{k: v.to(self.model.device) for k, v in t.items()})
        
        # print(model_output)
        embeddings = model_output.last_hidden_state[:, 0, :]
        # print(embeddings)
        embeddings = nn.functional.normalize(embeddings)
        # print(embeddings)
        return embeddings[0].cpu().numpy()