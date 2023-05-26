# pip install nltk rouge
# pip install scikit-learn
# pip install transformers

import json
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge 
from sklearn.metrics import f1_score
from collections import Counter

# model_name = "THUDM/chatglm-6b"
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()

# load test data
with open("psychology.json", "r") as file:
    test_data = json.load(file)

rouge = Rouge()
total_bleu = total_rouge_l = total_rouge_1 = total_rouge_2 = 0

for item in test_data:
    source_text = item['response']
    reference_text = item['human_answers']

    inputs = tokenizer.encode(source_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, temperature=0.7)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # BLEU-4
    reference = reference_text.split()
    candidate = generated_text.split()
    bleu_score = sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25))
    total_bleu += bleu_score

    # ROUGE
    scores = rouge.get_scores(generated_text, reference_text)
    total_rouge_1 += scores[0]['rouge-1']['f']
    total_rouge_2 += scores[0]['rouge-2']['f']
    total_rouge_l += scores[0]['rouge-l']['f']

# average point
average_bleu = total_bleu / len(test_data)
average_rouge_1 = total_rouge_1 / len(test_data)
average_rouge_2 = total_rouge_2 / len(test_data)
average_rouge_l = total_rouge_l / len(test_data)

print(f"Average Bleu-4 Score: {average_bleu}")
print(f"Average ROUGE-1: {average_rouge_1}")
print(f"Average ROUGE-2: {average_rouge_2}")
print(f"Average ROUGE-L: {average_rouge_l}")
