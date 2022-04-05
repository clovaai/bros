# BROS

## Introduction

BROS (BERT Relying On Spatiality) is a pre-trained language model focusing on text and layout for better key information extraction from documents.
Given the OCR results of the document image, which are text and bounding box pairs, it can perform various key information extraction tasks, such as extracting an ordered item list from receipts.
For more details, please refer to our paper:

**BROS: A Pre-trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents**<br> 
Teakgyu Hong, Donghyun Kim, Mingi Ji, Wonseok Hwang, Daehyun Nam, Sungrae Park<br>
AAAI 2022 - Main Technical Track

[[arXiv]](https://arxiv.org/abs/2108.04539)

## Pre-trained models

| name               | # params | Hugging Face - Models                                                                           |
|--------------------|---------:|-------------------------------------------------------------------------------------------------|
| bros-base-uncased  |   < 110M | [naver-clova-ocr/bros-base-uncased](https://huggingface.co/naver-clova-ocr/bros-base-uncased)   |
| bros-large-uncased |   < 340M | [naver-clova-ocr/bros-large-uncased](https://huggingface.co/naver-clova-ocr/bros-large-uncased) |

## Model usage

The example code below is written with reference to [LayoutLM](https://huggingface.co/docs/transformers/model_doc/layoutlm).

```python
import torch
from bros import BrosTokenizer, BrosModel


tokenizer = BrosTokenizer.from_pretrained("naver-clova-ocr/bros-base-uncased")
model = BrosModel.from_pretrained("naver-clova-ocr/bros-base-uncased")


width, height = 1280, 720

words = ["to", "the", "moon!"]
quads = [
    [638, 451, 863, 451, 863, 569, 638, 569],
    [877, 453, 1190, 455, 1190, 568, 876, 567],
    [632, 566, 1107, 566, 1107, 691, 632, 691],
]

bbox = []
for word, quad in zip(words, quads):
    n_word_tokens = len(tokenizer.tokenize(word))
    bbox.extend([quad] * n_word_tokens)

cls_quad = [0.0] * 8
sep_quad = [width, height] * 4
bbox = [cls_quad] + bbox + [sep_quad]

encoding = tokenizer(" ".join(words), return_tensors="pt")
input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]

bbox = torch.tensor([bbox])
bbox[:, :, [0, 2, 4, 6]] = bbox[:, :, [0, 2, 4, 6]] / width
bbox[:, :, [1, 3, 5, 7]] = bbox[:, :, [1, 3, 5, 7]] / height

outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask)
last_hidden_state = outputs.last_hidden_state

print("- last_hidden_state")
print(last_hidden_state)
print()
print("- last_hidden_state.shape")
print(last_hidden_state.shape)
```

Result
```
- last_hidden_state
tensor([[[-0.0342,  0.2487, -0.2819,  ...,  0.1495,  0.0218,  0.0484],
         [ 0.0792, -0.0040, -0.0127,  ..., -0.0918,  0.0810,  0.0419],
         [ 0.0808, -0.0918,  0.0199,  ..., -0.0566,  0.0869, -0.1859],
         [ 0.0862,  0.0901,  0.0473,  ..., -0.1328,  0.0300, -0.1613],
         [-0.2925,  0.2539,  0.1348,  ...,  0.1988, -0.0148, -0.0982],
         [-0.4160,  0.2135, -0.0390,  ...,  0.6908, -0.2985,  0.1847]]],
       grad_fn=<NativeLayerNormBackward>)

- last_hidden_state.shape
torch.Size([1, 6, 768])
```

## Fine-tuning examples

Please refer to [docs/finetuning_examples.md](docs/finetuning_examples.md).

## Acknowledgements

We referenced the code of [LayoutLM](https://huggingface.co/docs/transformers/model_doc/layoutlm) when implementing BROS in the form of Hugging Face - transformers.  
In this repository, we used two public benchmark datasets, [FUNSD](https://guillaumejaume.github.io/FUNSD/) and [SROIE](https://rrc.cvc.uab.es/?ch=13).

## License

```
Copyright 2022-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
