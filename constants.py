seed = 0

base_model_path = "/root/autodl-tmp/Qwen3-Reranker-4B"
pretrain_lora_path = None
lora_path = "/root/autodl-tmp/pseudo_lora"
use_gptq = "gptq" in base_model_path

positive = "Yes"
negative = "No"
judge_words = "Violation:"
system_prompt = '''You are given a comment from reddit and a rule. 
Your task is to classify whether the comment violates the rule. 
Only respond Yes/No.'''

frac = 0.05
use_train = True

# import kagglehub

# deterministic = kagglehub.package_import('wasupandceacar/deterministic').deterministic
# deterministic.init_all(seed)

