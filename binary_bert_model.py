from torch import nn
from transformers import AutoModel
from transformers import BertModel

class BinaryBertModel(nn.Module):
    
    def __init__(
        self, 
        seq_length=64,
        pretrained_bert_model="bert-base-uncased",
        dropout_p=0,
        ):
        super().__init__()
        
        self.pretrained_model = BertModel.from_pretrained(pretrained_bert_model)
        features_dim = self.pretrained_model.config.hidden_size
        
    
        self.flatten_layer = nn.Sequential(
            nn.Linear(features_dim, 1),
            nn.ReLU(),
        )
            
        self.hidden_layer = nn.Sequential(
            nn.Linear(seq_length, seq_length),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.LayerNorm(seq_length),
        )
            
        self.bin_layer = nn.Sequential(
            nn.Linear(seq_length, 1),
            nn.Sigmoid(),
        )
        
    
    def forward(self, tokens, attention_mask):
        tokens = self.pretrained_model(
            input_ids = tokens,
            attention_mask = attention_mask,
        )["last_hidden_state"]
        
        tokens = self.flatten_layer(tokens)
        tokens = tokens.squeeze(2)
        
        tokens = self.hidden_layer(tokens)
        return self.bin_layer(tokens)