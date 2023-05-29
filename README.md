# chatglm6b_ft

ChatGLM-6B local build and fine tuning||ChatGLM-6B本地化部署与微调

## 1. Deployment

ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 General Language Model (GLM) 架构，具有 62 亿参数。结合模型量化技术，用户可以在消费级的显卡上进行本地部署（INT4 量化级别下最低只需 6GB 显存）。 ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答。[清华Github源](https://github.com/THUDM/ChatGLM-6B)

### ① 本地部署

GPU:FP16（无量化）13 GB显存显卡

首先拷贝仓库，之后安装requirements.txt指示的package，运行[chatglm6b.py](code/chatglm6b.py)修改prompt部分获得response。

```
# git clone https://huggingface.co/THUDM/chatglm-6b
# pip install -qr requirements.txt

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()
response, history = model.chat(tokenizer, "谈7年的男友，分开两年又和好，嫌我胖是不是不爱我？", history=[])
print(response)
```

### ② Google Colaboratory部署

GPU: Tesla T4，需选择最少6 GB显存显卡

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-4UOCQtzX2OsdgbQOiukeX2r-wNCEJDC) 在线部署ChatGLM-6B的量化4等级模型，File path:`code/ChatGLM_6B_int4_Web_Demo.ipynb`

生成的界面示例如下：

![chatglm6b_colab_demo](photo/chatglm6b_colab_demo.JPG)

## 2. Fine-tuning

基于清华的 ChatGLM-6B + LoRA 进行finetune

### ① 概念介绍

LoRA: Low-Rank Adaptation of Large Language Models，直译为大语言模型的低阶适应，是一种PEFT（参数高效性微调方法）LoRA的基本原理是冻结预训练好的模型权重参数，在冻结原模型参数的情况下，通过往模型中加入额外的网络层，并只训练这些新增的网络层参数。由于这些新增参数数量较少，这样不仅 finetune 的成本显著下降，还能获得和全模型微调类似的效果。

Bleu-4: Bleu 是一种在文本生成中常用的评估指标，主要用来衡量机器翻译的结果和人工翻译的相似度。Bleu通过对比机器翻译结果和一个或多个人工翻译的结果，看看有多少n-gram（n个连续字或词的组合）是匹配的。这个比例越高，Bleu得分就越高，意味着机器翻译的质量越高。Bleu-4 取 4-gram 进行匹配。

Rouge: Rouge-1 与 Rouge-2 分别衡量一元与二元词组的匹配度，ROUGE-L 基于最长公共子序列计算。对于两个序列（这里的序列指的是词或者句子），最长公共子序列是指在所有的子序列中长度最长的一个。Rouge-l 可以很好地捕捉更长的词序列的匹配，能更好的评估完整性和连贯性。

### ② 数据集

训练集: Alpaca指令微调数据集: `dataset/alpaca_en`, Alpaca中文翻译数据集：`dataset/alpaca-chinese-dataset-main` 

取训练集数据大概总共有15万条，在单卡A100上运行大概需要14小时。

验证集：取Alibaba开源的DAMO_ConvAI中文数据集中的psychology部分：`dataset/DAMO_ConvAI`

### ③ 运行环境(GPU RAM ≥ 24GB)

| GPU in Colab    | 使用情况     |
| ------------- | ------------ |
| A100     |    ✔️    |
| V100 | ❌ |
| Tesla-T4  |  ❌ |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dH7QZyyzyG5YHw2FGFXpy3V8p0DxYucu#scrollTo=VLG3jYxUaZmg) File path: `code/chatglm_tuning.ipynb`

### ④ 关键参数

关键参数设置：
```
training_args = TrainingArguments(
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=32,   # batch size for evaluation
    learning_rate=2e-5,              # learning rate
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay  
)
```

LoRA微调核心代码：
```
model = ChatGLMForConditionalGeneration.from_pretrained(args.model_dir)
config = LoraConfig(r=args.lora_r,
                    lora_alpha=32,
                    target_modules=["query_key_value"],
                    lora_dropout=0.1,
                    bias="none",
                    task_type="CAUSAL_LM",
                    inference_mode=False,
                    )

model = get_peft_model(model, config)
```

### ⑤ 实验结果与分析

相较ChatGLM-6B微调前，微调后模型的BLEU-4，Rouge在上述验证集上的得分如下：

| Index    | Score %     |
| ------------- | ------------ |
| BLEU-4     |    16.68(+2. 01）    |
| Rouge-1 | 36.51(+1.53) |
| Rouge-2  |  15.31(+2.07)  |
| Rouge-l    |   26.31(+2.30)  |

LoRA微调权重保存在`output/chatglm_lora.pt`，Index测试代码：`code/validation_cn.py`，合并权重见`code/load_weight.py`，英文示例见`code/chatglm-tuning.ipynb`

实验结论：保持基础模型性能的同时，对中文语境下特定任务或领域产生更高质量的回答且未出现灾难性遗忘现象。



