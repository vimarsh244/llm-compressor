"""
Example script demonstrating Power-of-Two (PoT) quantization using llm-compressor.

This script shows how to apply PoT quantization to a LLaMA model, which constrains
all weights and activations to powers of two, enabling efficient inference using
bit-shift operations instead of multiplications.
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization.pot import PoTQuantizationModifier
from llmcompressor.utils import dispatch_for_generation

# Select model and load it
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

print(f"Loading model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Select calibration dataset
DATASET_ID = "mit-han-lab/pile-val-backup"
DATASET_SPLIT = "validation"

# Select number of samples for calibration
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

print(f"Loading calibration dataset: {DATASET_ID}")
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    """Preprocess the dataset examples."""
    return {
        "text": tokenizer.apply_chat_template(
            [{"role": "user", "content": example["text"]}],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


def tokenize(sample):
    """Tokenize the input samples."""
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


print("Configuring PoT quantization...")

# Configure PoT quantization
# This will quantize weights to 4-bit PoT and activations to 8-bit PoT
pot_recipe = PoTQuantizationModifier(
    targets=["Linear"],
    ignore=["lm_head"],  # typically keep the output layer at full precision
    pot_bits=4,  # use 4-bit PoT quantization for weights
    scheme={
        "weights": {
            "num_bits": 4,
            "type": "int",
            "symmetric": True,
            "strategy": "channel",  # per-channel quantization for better accuracy
        },
        "input_activations": {
            "num_bits": 8,
            "type": "int", 
            "symmetric": True,
            "strategy": "tensor",  # per-tensor quantization for activations
        }
    }
)

print("Applying PoT quantization...")

# Apply PoT quantization
oneshot(
    model=model,
    dataset=ds,
    recipe=pot_recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("PoT quantization completed!")

# Test the quantized model
print("\n" + "="*50)
print("TESTING QUANTIZED MODEL GENERATION")
print("="*50)

dispatch_for_generation(model)
input_ids = tokenizer("Hello, my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=100, do_sample=False)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")

print("="*50 + "\n")

# Save the quantized model
SAVE_DIR = MODEL_ID.split("/")[-1] + "-pot-w4a8"
print(f"Saving quantized model to: {SAVE_DIR}")
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print("Model saved successfully!")
print(f"\nTo load the quantized model later:")
print(f"model = AutoModelForCausalLM.from_pretrained('{SAVE_DIR}')")
print(f"tokenizer = AutoTokenizer.from_pretrained('{SAVE_DIR}')")