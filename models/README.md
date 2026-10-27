Hugging Face models' config and tokenizer files, never weights (the simulator builds models on the meta device), one directory per `<org>--<name>/<commit>/`. Not tracked. To fetch one (`pip install huggingface_hub`; the commit is on the model page's history):

```bash
hf download Qwen/Qwen2.5-7B-Instruct --revision <commit> \
  --exclude "*.safetensors" --exclude "*.bin" --exclude "*.pt" --exclude "*.gguf" \
  --local-dir models/Qwen--Qwen2.5-7B-Instruct/<commit>
```
