# Language Generation

## Example usage:
To test the fine-tuned GPT-J model, this script can be issued:
```script
python run_generation.py \
    --model_name_or_path=finetuned
    --length 512
```

## Generate code with fine-tuned model
To test the fine-tuned GPT-J model, this script can be issued:
```script
python run_generate.py --model_name_or_path=finetuned --length 512
```

Alternatively, the model can be used in custom code like this to generate text in batches:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("finetuned")
tokenizer.pad_token = "[PAD]"

# Load model
model = AutoModelForCausalLM.from_pretrained("finetuned").to(device)
print("Model loaded")

prompts = ["contract HelloWorld {", "function hello() public"]

# Tokenize
input_ids = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

# Generate
gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=512,
)
generated_texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
print(generated_texts)
```

## Serving

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/andstor/17105942ecf864d1796c36f6e74c5f29/andstor-gpt-j-6b-smart-contract-server.ipynb)

```script
transformers-cli serve --task --device 0 feature-extraction --model andstor/gpt-j-6B-smart-contract --config andstor/gpt-j-6B-smart-contract --tokenizer andstor/gpt-j-6B-smart-contract
```

```script
curl -X POST "http://localhost:8888/forward" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"inputs\":\"Hello world!\"}"
```