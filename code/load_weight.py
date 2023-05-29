# load fine tuning weights.
import torch
from transformers import AutoTokenizer, AutoModel

torch.load("chatglm-lora.pt")
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()

# Load the weights
model.load_state_dict(torch.load("chatglm-lora.pt"))
