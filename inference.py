import os
os.environ["VLLM_USE_V1"] = "0"

import random
import vllm
import torch
import numpy as np
import pandas as pd
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
from vllm.lora.request import LoRARequest
from utils import build_dataset
from constants import *
import multiprocessing as mp

def run_inference_on_device(df_slice):
    llm = vllm.LLM(
        base_model_path,
        quantization="gptq" if use_gptq else None,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.98,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=2048,
        disable_log_stats=True,
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=64,
    )
    tokenizer = llm.get_tokenizer()
    outputs = llm.generate(
        build_dataset(df_slice)["prompt"],
        vllm.SamplingParams(
            skip_special_tokens=True,
            max_tokens=1,
            logits_processors=[MultipleChoiceLogitsProcessor(tokenizer, choices=[positive, negative])],
            logprobs=2,
        ),
        use_tqdm=True,
        lora_request=LoRARequest("lora1", 1, lora_path)
    )
    log_probs = [{lp.decoded_token: np.exp(lp.logprob) for lp in out.outputs[0].logprobs[0].values()} for out in outputs]
    predictions = pd.DataFrame(log_probs)[[positive, negative]]
    predictions["row_id"] = df_slice["row_id"].values
    return predictions

def worker(device_id, df_slice, return_dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    print(f"[Worker {device_id}] Running on GPU {device_id}, data size={len(df_slice)}")
    preds = run_inference_on_device(df_slice)
    return_dict[device_id] = preds

def main():
    test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
    test_df["positive_example"] = test_df.apply(lambda row: random.choice([row["positive_example_1"], row["positive_example_2"]]), axis=1)
    test_df["negative_example"] = test_df.apply(lambda row: random.choice([row["negative_example_1"], row["negative_example_2"]]), axis=1)
    test_df = test_df.drop(columns=["positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"], errors="ignore")

    mid = len(test_df) // 2
    df0 = test_df.iloc[:mid].reset_index(drop=True)
    df1 = test_df.iloc[mid:].reset_index(drop=True)

    manager = mp.Manager()
    return_dict = manager.dict()
    p0 = mp.Process(target=worker, args=(0, df0, return_dict))
    p1 = mp.Process(target=worker, args=(1, df1, return_dict))
    p0.start()
    p1.start()
    p0.join()
    p1.join()

    predictions = pd.concat([return_dict[0], return_dict[1]], ignore_index=True)
    submission = predictions[["row_id", positive]].rename(columns={positive: "rule_violation"})
    submission.to_csv("/kaggle/working/submission.csv", index=False)

if __name__ == "__main__":
    main()