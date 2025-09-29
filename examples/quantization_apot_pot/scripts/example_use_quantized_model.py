"""Example of how to use quantized models in your own applications."""

from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.utils.dev import dispatch_for_generation


class QuantizedModelInference:
    """Simple wrapper class for quantized model inference."""
    
    def __init__(self, model_path: str):
        """Initialize the quantized model."""
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = None
        self.load_model()
    
    def load_model(self):
        """Load the quantized model and tokenizer."""
        print(f"Loading quantized model from: {self.model_path}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Important: dispatch for generation with quantized models
        dispatch_for_generation(self.model)
        
        self.device = self.model.device
        print(f"Model loaded successfully on device: {self.device}")
    
    def generate_text(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7, top_p: float = 0.9):
        """Generate text from a prompt."""
        
        # Tokenize input
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).input_ids.to(self.device)
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode and return generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Remove the input prompt from the generated text
        if generated_text.startswith(prompt):
            return generated_text[len(prompt):].strip()
        else:
            return generated_text
    
    def chat(self, message: str):
        """Simple chat interface."""
        return self.generate_text(message, max_tokens=150)
    
    def cleanup(self):
        """Clean up GPU memory."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def demo_apot_model():
    """Demonstrate using APoT quantized model."""
    model_path = "TinyLlama-1.1B-Chat-v1.0-apot-w4a8-t2"
    
    if not Path(model_path).exists():
        print(f"APoT model not found at: {model_path}")
        print("Please run: python llama_apot_example.py")
        return
    
    print("=" * 60)
    print("APOT MODEL DEMO")
    print("=" * 60)
    
    # Initialize model
    model = QuantizedModelInference(model_path)
    
    # Test various generations
    test_prompts = [
        "Explain machine learning in simple terms:",
        "Write a Python function to reverse a string:",
        "What are the advantages of quantized neural networks?",
        "Tell me about the weather today:"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        response = model.generate_text(prompt, max_tokens=80)
        print(f"Response: {response}")
        print("-" * 40)
    
    # Cleanup
    model.cleanup()


def demo_pot_model():
    """Demonstrate using PoT quantized model."""
    model_path = "TinyLlama-1.1B-Chat-v1.0-pot-w4a8"
    
    if not Path(model_path).exists():
        print(f"PoT model not found at: {model_path}")
        print("Please run: python llama_pot_example.py")
        return
    
    print("=" * 60)
    print("POT MODEL DEMO")
    print("=" * 60)
    
    # Initialize model
    model = QuantizedModelInference(model_path)
    
    # Interactive chat demo
    print("\nInteractive chat demo (type 'quit' to exit):")
    print("-" * 40)
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_input:
            response = model.chat(user_input)
            print(f"Bot: {response}")
    
    # Cleanup
    model.cleanup()


def compare_models():
    """Compare APoT and PoT models side by side."""
    apot_path = "TinyLlama-1.1B-Chat-v1.0-apot-w4a8-t2"
    pot_path = "TinyLlama-1.1B-Chat-v1.0-pot-w4a8"
    
    # Check if both models exist
    if not Path(apot_path).exists():
        print(f"APoT model not found: {apot_path}")
        return
    
    if not Path(pot_path).exists():
        print(f"PoT model not found: {pot_path}")
        return
    
    print("=" * 80)
    print("SIDE-BY-SIDE MODEL COMPARISON")
    print("=" * 80)
    
    # Load both models
    apot_model = QuantizedModelInference(apot_path)
    pot_model = QuantizedModelInference(pot_path)
    
    comparison_prompts = [
        "Explain the concept of quantization:",
        "Write a simple algorithm to sort numbers:",
        "What is the future of AI?"
    ]
    
    for prompt in comparison_prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 60)
        
        apot_response = apot_model.generate_text(prompt, max_tokens=100)
        pot_response = pot_model.generate_text(prompt, max_tokens=100)
        
        print(f"APoT Response: {apot_response}")
        print(f"PoT Response:  {pot_response}")
        print("=" * 60)
    
    # Cleanup
    apot_model.cleanup()
    pot_model.cleanup()


def main():
    """Main demo function."""
    print("QUANTIZED MODEL USAGE EXAMPLES")
    print("=" * 80)
    
    while True:
        print("\nChoose an option:")
        print("1. Demo APoT model")
        print("2. Demo PoT model (interactive)")
        print("3. Compare both models")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            demo_apot_model()
        elif choice == "2":
            demo_pot_model()
        elif choice == "3":
            compare_models()
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()
