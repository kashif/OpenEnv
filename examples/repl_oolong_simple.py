#!/usr/bin/env python3
"""
Simple REPL + Oolong example with recursive LLM calls (RLM paradigm).

This connects to the REPL Space which has llm_query and llm_batch enabled
via HuggingFace Inference API.

Usage:
    python examples/repl_oolong_simple.py
"""
from __future__ import annotations

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from repl_env import REPLEnv
from repl_env.prompts import (
    RLM_SYSTEM_PROMPT,
    build_initial_prompt,
    extract_code_blocks,
    format_observation,
)

# ============== CONFIGURATION ==============
SPACE_URL = "https://sergiopaniego-repl.hf.space"
MODEL_NAME = "Qwen/Qwen3-1.7B"
DATASET_SUBSET = "toy_dnd"
DATASET_SPLIT = "validation"
EXAMPLE_INDEX = 0
MAX_ITERATIONS = 15
# ===========================================


def main():
    print("=" * 60)
    print("REPL + Oolong with Recursive LLM Calls (RLM)")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading dataset example {EXAMPLE_INDEX}...")
    dataset = load_dataset("oolongbench/oolong-real", DATASET_SUBSET, split=DATASET_SPLIT)
    example = dataset[EXAMPLE_INDEX]

    context = example["context_window_text"]
    question = example["question"]
    expected = str(example["answer"])

    print(f"Question: {question}")
    print(f"Expected answer: {expected}")
    print(f"Context length: {len(context):,} chars")

    # Load model for the outer loop (agent)
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )

    def llm_chat(messages: list[dict]) -> str:
        """LLM function for chat-style messages (outer loop)."""
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
        )
        return tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

    # Build task prompt - emphasize llm_query and llm_batch availability
    task_prompt = f"""Answer the following question based on the context provided.

Question: {question}

Instructions:
- The context is available in the 'context' variable (a string with {len(context):,} chars)
- You have access to llm_query(prompt) to ask the LLM questions about text chunks
- You have access to llm_batch(prompts_list) to process multiple prompts efficiently
- Strategy: Split the context into manageable chunks and use llm_query/llm_batch to analyze them
- When you find the answer, use: print(f'FINAL(your_answer)')

Example approach:
```python
# Split context into chunks
chunks = [context[i:i+5000] for i in range(0, len(context), 5000)]

# Ask LLM to count something in each chunk
results = llm_batch([f"Count X in this text: {{chunk}}" for chunk in chunks])

# Aggregate results
total = sum(extract_number(r) for r in results)
print(f'FINAL({{total}})')
```
"""

    # Connect to REPL Space (which now has llm_query/llm_batch via HF Inference API)
    print(f"\nConnecting to: {SPACE_URL}")
    print("Note: The Space now has llm_query/llm_batch enabled via HF Inference API")
    
    with REPLEnv(base_url=SPACE_URL) as env:
        # Reset with context
        result = env.reset(context=context, task_prompt=task_prompt, max_iterations=MAX_ITERATIONS)
        obs = result.observation

        print(f"Context loaded: {obs.context_length:,} chars")
        print(f"Available variables: {obs.available_variables}")

        # Build initial messages
        messages = [
            {"role": "system", "content": RLM_SYSTEM_PROMPT},
            {"role": "user", "content": build_initial_prompt(
                task_prompt=task_prompt,
                context_length=obs.context_length,
                context_preview=obs.context_preview,
                variables=obs.available_variables,
            )},
        ]

        # RLM loop
        final_answer = None
        for i in range(1, MAX_ITERATIONS + 1):
            print(f"\n--- Iteration {i} ---")

            response = llm_chat(messages)
            print(f"LLM: {response[:400]}{'...' if len(response) > 400 else ''}")

            code_blocks = extract_code_blocks(response)
            if not code_blocks:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "Please provide Python code in ```python``` blocks."})
                continue

            for code in code_blocks:
                print(f"\nExecuting:\n{code[:300]}{'...' if len(code) > 300 else ''}")

                result = env.execute(code)
                obs = result.observation

                print(f"Success: {obs.result.success}")
                if obs.result.stdout:
                    print(f"Output: {obs.result.stdout[:300]}{'...' if len(obs.result.stdout) > 300 else ''}")
                if obs.result.stderr:
                    print(f"Stderr: {obs.result.stderr[:200]}")

                if result.done:
                    state = env.state()
                    final_answer = state.final_answer if state else obs.metadata.get("final_answer")
                    break

            if final_answer is not None:
                break

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": format_observation(obs)})

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Question: {question}")
    print(f"Expected: {expected}")
    print(f"Got:      {final_answer}")

    if final_answer and str(final_answer).strip().lower() == expected.strip().lower():
        print("✓ CORRECT!")
    else:
        print("✗ INCORRECT")


if __name__ == "__main__":
    main()
