from vllm import LLM

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


model = LLM("TinyLlama-1.1B-Chat-v1.0-apot-4term-w8a8")
for prompt in TEST_PROMPTS:
    output = model.generate(prompt)
    print(f"Prompt: {prompt}\nGenerated: {output}\n{'-'*40}\n")


