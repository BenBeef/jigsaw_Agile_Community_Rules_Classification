"""
使用预训练的 Llama-3.1-8B-Instruct 模型进行问答
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型路径
MODEL_PATH = "/root/autodl-tmp/warm-start__grpo__think__Llama-3.1-8B-Instruct"

print("正在加载模型和分词器...")
print(f"模型路径: {MODEL_PATH}")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,  # 使用 bfloat16 以节省显存
    device_map="auto",  # 自动分配到可用的 GPU
    low_cpu_mem_usage=True
)

print("✓ 模型加载完成！")
print(f"模型设备: {model.device}")
print(f"模型参数量: {model.num_parameters() / 1e9:.2f}B")


def chat(user_message, system_prompt="You are an excellent content moderation expert.", max_new_tokens=1024, temperature=0.1, print_tokens=True):
    """
    使用 Llama-3.1 模型进行对话
    
    参数:
        user_message: 用户的问题
        system_prompt: 系统提示词
        max_new_tokens: 最大生成token数
        temperature: 生成温度，控制随机性
        print_tokens: 是否打印token数量统计
    
    返回:
        模型的回答
    """
    # 构建 Llama-3.1 格式的对话消息
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    # 使用分词器的 chat template 进行格式化
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 记录开始时间
    start_time = time.time()
    
    # 编码输入
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # 获取输入token数量
    input_token_count = inputs['input_ids'].shape[1]
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 记录结束时间
    elapsed_time = time.time() - start_time
    
    # 获取输出token数量（只计算生成的部分）
    output_token_count = outputs.shape[1] - input_token_count
    total_token_count = outputs.shape[1]
    
    # 计算速度（tokens per second）
    tokens_per_second = output_token_count / elapsed_time if elapsed_time > 0 else 0
    
    if print_tokens:
        print(f"📊 Token统计 | 输入: {input_token_count} | 输出: {output_token_count} | 总计: {total_token_count} | 耗时: {elapsed_time:.2f}s | 速度: {tokens_per_second:.2f} tokens/s")
    
    # 解码输出（只返回生成的部分）
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    return generated_text


text = '''
You are given a community rule, several example cases with their judgments, and a user query to evaluate.

Your task: Determine whether the user query violates the given rule.

Instructions:
- Each example case contains an <output> tag with the judgment: "yes" means it violates the rule, "no" means it does not.
- Use these example cases as references to guide your judgment of the query.
- Provide your answer in the same format: <output>yes</output> or <output>no</output>

Rule:
No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.

=======================================================================================================
cases

case1：
"
If you could tell your younger self something different about sex, what would that be?

i AM IN A CONTEST TO WIN FUNDING FOR MY SEX POSITIVE FILM: VOTE HERE:

http://sheknows.offerpop.com/campaign/813112/entry/v144417
"

<output> yes </output>


case2:
"
Watch Golden Globe Awards 2017 Live Online in HD Coverage without ADS (VIP STREAMS)
=

HD STREAM QUALITY >>> [WATCH LINK1](http://forum.submitexpress.com/viewtopic.php?f=9&t=215858)
=

HD BROADCASTING QUALITY >>> [WATCH LINK1](http://forum.submitexpress.com/viewtopic.php?f=9&t=215858)
=

Mobile Compatibility: YES
=

NO ADS | NO ADS | ADS
=
"

<output> no <ouput>

=======================================================================================================
query:
Banks don't want you to know this! Click here to know more!

Does This body violate the rule
'''

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Llama-3.1-8B-Instruct 中文问答系统")
    print("="*60 + "\n")
    
    # 示例问题
    questions = [
        "你好！请用中文介绍一下你自己。",
        "你好！请用中文介绍一下你自己。",
        text,
        text,
        text,
        text,
        text,
        text,
    ]
    
    # 串行处理所有问题（GPU模型不支持真正的并发）
    print(f"🚀 开始处理 {len(questions)} 个问题...")
    total_start = time.time()
    
    for i, question in enumerate(questions, 1):
        print(f"\n【问题 {i}/{len(questions)}】开始处理...")
        if len(question) > 100:
            print(f"内容预览: {question[:100]}...")
        else:
            print(f"内容: {question}")
        print("-" * 60)
        answer = chat(question)
        print(f"【回答】{answer}")
        print("=" * 60)
    
    total_time = time.time() - total_start
    print(f"\n✅ 所有问题处理完成！总耗时: {total_time:.2f}秒 | 平均每题: {total_time/len(questions):.2f}秒")
    
