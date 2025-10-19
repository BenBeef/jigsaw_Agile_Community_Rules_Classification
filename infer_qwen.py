import os
import pandas as pd
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
import torch
import vllm
import numpy as np
from vllm.lora.request import LoRARequest
import argparse
from scipy.special import softmax
df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

MODEL_NAME = "/kaggle/input/qwen2.5/transformers/14b-instruct-gptq-int4/1"
LORA_PATH = "/kaggle/input/lora_14b_gptq_1epoch_r32/keras/default/1"
if __name__=='__main__':
    os.environ["VLLM_USE_V1"] = "0"

    llm = vllm.LLM(
        MODEL_NAME,
        # quantization='awq',
        quantization='gptq',
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.98,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=2836,
        disable_log_stats=True,
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=32
    )
    tokenizer = llm.get_tokenizer()
    SYS_PROMPT = """
You are given a comment on reddit. Your task is to classify if it violates the given rule. Only respond Yes/No.
"""
    
    prompts = []
    for i, row in df.iterrows():
        text = f"""
    r/{row.subreddit}
    Rule: {row.rule}
    
    1) {row.positive_example_1}
    Violation: Yes
    
    2) {row.positive_example_2}
    Violation: Yes
    
    3) {row.negative_example_1}
    Violation: No
    
    4) {row.negative_example_2}
    Violation: No
    
    5) {row.body}
    """
        
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": text}
        ]
    
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        ) + "Answer:"
        prompts.append(prompt)
    
    df["prompt"] = prompts
    
    mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=['Yes','No'])
    outputs = llm.generate(
        prompts,
        vllm.SamplingParams(
            skip_special_tokens=True,
            max_tokens=1,
            logits_processors=[mclp],
            logprobs=2,
        ),
        use_tqdm=True,
        lora_request=LoRARequest("default", 1, LORA_PATH)
    )
    logprobs = [
        {lp.decoded_token: lp.logprob for lp in out.outputs[0].logprobs[0].values()}
        for out in outputs
    ]
    logit_matrix = pd.DataFrame(logprobs)[['Yes','No']]
    df = pd.concat([df, logit_matrix], axis=1)
    
    df[['Yes',"No"]] = df[['Yes',"No"]].apply(lambda x: softmax(x.values), axis=1, result_type="expand")
    df["pred"] = df["Yes"]
    df['rule_violation'] = df["pred"]
    df[['row_id', 'rule_violation']].to_csv("submission_qwen14b.csv",index=False)
    pd.read_csv('submission_qwen14b.csv')

