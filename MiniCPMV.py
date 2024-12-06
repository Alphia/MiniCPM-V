import json

import torch
from transformers import AutoModel, AutoTokenizer


class MiniCPMV26:
    def __init__(self, model_path, multi_gpus=False) -> None:
        print('torch_version:', torch.__version__)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation='sdpa',
                                               torch_dtype=torch.bfloat16)
        self.model.eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def chat(self, msgs, **kwargs):
        return self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            **kwargs
        )


class MiniCPMV25:
    def __init__(self, model_path) -> None:
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval().cuda()

    def chat(self, rgb_img, prompt, sampling=False, temperature=0.7):
        return self.model.chat(
            image=rgb_img,
            msgs=json.loads(prompt),
            tokenizer=self.tokenizer,
            sampling=sampling,
            temperature=temperature
        )


class MiniCPMV25INT4:
    def __init__(self, model_path) -> None:
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()

    def chat(self, rgb_img, prompt):
        return self.model.chat(
            image=rgb_img,
            msgs=json.loads(prompt),
            tokenizer=self.tokenizer,
            sampling=True,
            temperature=0.7
        )
