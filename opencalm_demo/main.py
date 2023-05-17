from accelerate.utils import write_basic_config
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    write_basic_config()
#@title モデルの読み込み
    model_id = "cyberagent/open-calm-3b" #@param ["cyberagent/open-calm-7b","cyberagent/open-calm-3b","cyberagent/open-calm-1b","cyberagent/open-calm-large","cyberagent/open-calm-medium","cyberagent/open-calm-small"]

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    instruct = "AIは今日目覚ましい" #@param {type: "string"}

    inputs = tokenizer(instruct, return_tensors="pt").to(model.device)

    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )

    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    print(output)

if __name__ == '__main__':
    main()
