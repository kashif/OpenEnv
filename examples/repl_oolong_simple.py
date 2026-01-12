#!/usr/bin/env python3
"""
Simple REPL + Oolong example with recursive LLM calls (RLM paradigm).

Demonstrates the unified REPLEnv API that works with both remote servers
and local execution using the same interface.

Usage:
    # Run against remote server
    python examples/repl_oolong_simple.py

    # Run locally (set SPACE_URL = None in the script)
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
# Set to None to run locally, or a URL to connect to remote Space
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
            enable_thinking=True,  # Enable Qwen3 thinking mode for better reasoning
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,  # Increased for thinking mode
            do_sample=True,
            top_k=50,
            top_p=0.9,
        )
        return tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

    # Build task prompt
    task_prompt = f"""Answer the following question about the context.

Question: {question}

The context ({len(context):,} chars) appears to be a D&D game transcript. Explore it first
to understand how dice rolls are represented, then count them accurately.
"""

    # Create environment - unified API for both local and remote!
    if SPACE_URL:
        print(f"\nConnecting to: {SPACE_URL}")
        env = REPLEnv(base_url=SPACE_URL)
    else:
        print("\nRunning locally")
        # For local mode, provide LLM functions for llm_query/llm_batch support
        def local_llm_query(prompt: str) -> str:
            return llm_chat([{"role": "user", "content": prompt}])

        def local_llm_batch(prompts: list[str]) -> list[str]:
            return [local_llm_query(p) for p in prompts]

        env = REPLEnv(llm_query_fn=local_llm_query, llm_batch_fn=local_llm_batch)

    # Reset environment - same API for both local and remote
    result = env.reset(context=context, task_prompt=task_prompt, max_iterations=MAX_ITERATIONS)
    obs = result.observation

    print(f"Context loaded: {obs.context_length:,} chars")
    print(f"Available variables: {obs.available_variables}")

    # Build initial messages for the agent
    context_preview = context[:500] + "..." if len(context) > 500 else context
    messages = [
        {"role": "system", "content": RLM_SYSTEM_PROMPT},
        {"role": "user", "content": build_initial_prompt(
            task_prompt=task_prompt,
            context_length=obs.context_length,
            context_preview=context_preview,
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

            # Execute code - same API for both local and remote!
            result = env.execute(code)
            obs = result.observation

            print(f"Success: {obs.result.success}")
            if obs.result.stdout:
                print(f"Output: {obs.result.stdout[:300]}{'...' if len(obs.result.stdout) > 300 else ''}")
            if obs.result.stderr:
                print(f"Stderr: {obs.result.stderr[:200]}")

            if result.done:
                state = env.state()
                final_answer = state.final_answer
                break

        if final_answer is not None:
            break

        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": format_observation(obs)})

    # Cleanup
    env.close()

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
