# validation_cn
# pip install nltk jiwer jieba

import json
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge 
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# load model and tokenizer
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()

# add test data
with open("psychology.json", "r") as file:
    test_data = json.load(file)

rouge = Rouge()
total_bleu = total_rouge_l = total_rouge_1 = total_rouge_2 = 0

for item in test_data:
    source_text = item['source']
    reference_text = item['target']

    # generate output
    inputs = tokenizer.encode(source_text, return_tensors="pt")
    outputs = model.response(inputs, max_length=150, num_return_sequences=1, temperature=0.7)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 计算BLEU-4分数
    reference = list(reference_text)
    candidate = list(generated_text)
    try:
        bleu_score = sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25))
    except ZeroDivisionError:
        bleu_score = 0
    total_bleu += bleu_score

    # 计算ROUGE分数
    scores = rouge.get_scores(generated_text, reference_text)
    total_rouge_l += scores[0]['rouge-l']['f']
    total_rouge_1 += scores[0]['rouge-1']['f']
    total_rouge_2 += scores[0]['rouge-2']['f']

# 计算平均得分
average_bleu = total_bleu / len(test_data)
average_rouge_l = total_rouge_l / len(test_data)
average_rouge_1 = total_rouge_1 / len(test_data)
average_rouge_2 = total_rouge_2 / len(test_data)

print(f"Average Bleu-4 Score: {average_bleu}")
print(f"Average ROUGE-L: {average_rouge_l}")
print(f"Average ROUGE-1: {average_rouge_1}")
print(f"Average ROUGE-2: {average_rouge_2}")
