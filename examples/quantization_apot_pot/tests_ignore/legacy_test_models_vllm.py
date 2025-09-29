from vllm import LLM
from vllm import SamplingParams
import time

# Test prompts - varied to test different aspects
TEST_PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "Write a short story about a robot learning to paint:",
    "Explain quantum computing in simple terms:",
    "What are the benefits of renewable energy?",
    "Describe the process of photosynthesis:",
    "Tell me a joke about programming:",
    "What is the meaning of life?",
    "How do neural networks work?",
    "Write a recipe for chocolate cake:"
]


# model = LLM("TinyLlama-1.1B-Chat-v1.0-apot-4term-w8a8")
model = LLM ("TinyLlama-1.1B-Chat-v1.0-pot-w4-weight-only")
for prompt in TEST_PROMPTS:
    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=0.7,
        top_p=0.9,
        n=1
    )

    # Generate without progress bars and extract only the text
    results = model.generate(prompt, sampling_params=sampling_params, use_tqdm=False)
    output = ""
    if results and results[0].outputs:
        # Join all candidate completions (if any)
        output = "\n".join(comp.text for comp in results[0].outputs)
    print(f"Prompt: {prompt}\nGenerated: {output}\n{'-'*40}\n")
    time.sleep(2)


