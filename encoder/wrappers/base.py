from transformers import AutoTokenizer, AutoModel


class BaseWrapper:
    
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)