# chatglm6b localisation
# first run: git clone https://huggingface.co/THUDM/chatglm-6b
# then run: pip install -qr requirements.txt
# last run: python3 chatglm6b.py
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()
response, history = model.chat(tokenizer, "谈7年的男友，分开两年又和好，嫌我胖是不是不爱我？", history=[])
print(response)