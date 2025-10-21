"""
使用 vLLM 加载 Llama-3.1-8B-Instruct 模型进行高效批量推理
vLLM 提供更高的吞吐量和真正的并发处理能力
"""

import time
from vllm import LLM, SamplingParams

# 模型路径
MODEL_PATH = "/root/autodl-tmp/warm-start__grpo__think__Llama-3.1-8B-Instruct"

print("正在使用 vLLM 加载模型...")
print(f"模型路径: {MODEL_PATH}")

# 初始化 vLLM
llm = LLM(
    model=MODEL_PATH,
    dtype="bfloat16",  # 使用 bfloat16
    trust_remote_code=True,
    gpu_memory_utilization=0.9,  # 使用90%的GPU显存
    max_model_len=8192,  # 最大序列长度
)

print("✓ 模型加载完成！")

# 获取tokenizer用于统计
tokenizer = llm.get_tokenizer()


def format_prompt(user_message, system_prompt="You are an excellent content moderation expert."):
    """
    格式化为 Llama-3.1 的对话格式
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    # 使用tokenizer的chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return formatted_prompt


def batch_chat(questions_list, system_prompt="You are an excellent content moderation expert.", 
               max_new_tokens=1024, temperature=0.1, print_stats=True):
    """
    使用 vLLM 批量处理多个问题（真正的并发）
    
    参数:
        questions_list: 问题列表
        system_prompt: 系统提示词
        max_new_tokens: 最大生成token数
        temperature: 生成温度
        print_stats: 是否打印统计信息
    
    返回:
        结果列表
    """
    print(f"🚀 准备批量处理 {len(questions_list)} 个问题...\n")
    
    # 格式化所有prompt
    prompts = [format_prompt(q, system_prompt) for q in questions_list]
    
    # 计算输入token数
    input_token_counts = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_token_counts.append(len(tokens))
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_new_tokens,
    )
    
    # 批量生成（vLLM会自动优化批处理）
    print("⚡ 开始批量生成...")
    start_time = time.time()
    
    outputs = llm.generate(prompts, sampling_params)
    
    elapsed_time = time.time() - start_time
    
    # 处理结果
    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        output_tokens = len(output.outputs[0].token_ids)
        input_tokens = input_token_counts[i]
        
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        
        results.append({
            'question': questions_list[i],
            'answer': generated_text,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
        })
        
        if print_stats:
            print(f"\n【问题 {i+1}/{len(questions_list)}】")
            if len(questions_list[i]) > 100:
                print(f"内容预览: {questions_list[i][:100]}...")
            else:
                print(f"内容: {questions_list[i]}")
            print("-" * 60)
            print(f"📊 Token | 输入: {input_tokens} | 输出: {output_tokens}")
            print(f"【回答】{generated_text}")
            print("=" * 60)
    
    # 总体统计
    total_tokens = total_input_tokens + total_output_tokens
    throughput = total_output_tokens / elapsed_time
    avg_time_per_question = elapsed_time / len(questions_list)
    
    print(f"\n✅ 所有问题处理完成！")
    print(f"📊 总统计:")
    print(f"   - 总耗时: {elapsed_time:.2f}秒")
    print(f"   - 平均每题: {avg_time_per_question:.2f}秒")
    print(f"   - 输入Token总数: {total_input_tokens}")
    print(f"   - 输出Token总数: {total_output_tokens}")
    print(f"   - Token总数: {total_tokens}")
    print(f"   - 吞吐量: {throughput:.2f} tokens/s")
    print(f"   - 平均生成速度: {total_output_tokens/elapsed_time:.2f} tokens/s")
    
    return results


# 测试文本
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
    print("Llama-3.1-8B-Instruct vLLM 批量推理系统")
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

    questions = questions + questions
    
    # 批量处理所有问题
    results = batch_chat(questions)

