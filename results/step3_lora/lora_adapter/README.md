# Instant-1.7B LoRA adapter (first run, not yet benchmarked)

This is a LoRA adapter for [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B). It was trained to answer directly, with thinking OFF and no hidden reasoning, using the model's own verified answers. Training details are in [../README.md](../README.md).

**Modification notice:** this adapter modifies the weights of LiquidAI/LFM3-1.7B. It is distributed under the LFM Open License v1.0; a copy is in `LICENSE`. Organisations with $10M or more in annual revenue need a commercial licence from Liquid AI (sales@liquid.ai). This project is not affiliated with Liquid AI.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("LiquidAI/LFM3-1.7B", trust_remote_code=True, dtype="auto")
model = PeftModel.from_pretrained(base, "lora_adapter").merge_and_unload()
tok = AutoTokenizer.from_pretrained("LiquidAI/LFM3-1.7B", trust_remote_code=True)
prompt = tok.apply_chat_template([{"role": "user", "content": "Reverse a string in Python."}],
                                 tokenize=False, add_generation_prompt=True, enable_thinking=False)
```
