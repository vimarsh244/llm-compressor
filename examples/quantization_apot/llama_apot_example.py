"""
Example script demonstrating Additive Power-of-Two (APoT) quantization using llm-compressor.

This script shows how to apply APoT quantization to a LLaMA model, which represents
values as sums of signed powers of two, providing better accuracy than simple PoT
while maintaining hardware efficiency.
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization.apot import APoTQuantizationModifier
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


print("Configuring APoT quantization...")

# Configure APoT quantization using config_groups
# This will quantize weights to 4-bit APoT with 2 terms and activations to 8-bit APoT
apot_recipe = APoTQuantizationModifier(
    config_groups={
        "group_0": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="channel",  # per-channel quantization for better accuracy
                observer="apot",
                observer_kwargs={"num_terms": 2},
            ),
            input_activations=QuantizationArgs(
                num_bits=8,
                type="int",
                symmetric=True,
                strategy="tensor",  # per-tensor quantization for activations
                observer="apot",
                observer_kwargs={"num_terms": 2},
            )
        )
    },
    ignore=["lm_head"],  # typically keep the output layer at full precision
    apot_bits=4,  # use 4-bit APoT quantization for weights
    num_terms=2,  # use 2-term APoT (sum of 2 signed powers of two)
)

print(f"Applying APoT quantization with {apot_recipe.num_terms} terms...")

# Apply APoT quantization
oneshot(
    model=model,
    dataset=ds,
    recipe=apot_recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("APoT quantization completed!")

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
SAVE_DIR = MODEL_ID.split("/")[-1] + f"-apot-{apot_recipe.num_terms}term-w4a8"
print(f"Saving quantized model to: {SAVE_DIR}")
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print("Model saved successfully!")
print(f"\nTo load the quantized model later:")
print(f"model = AutoModelForCausalLM.from_pretrained('{SAVE_DIR}')")
print(f"tokenizer = AutoTokenizer.from_pretrained('{SAVE_DIR}')")

print(f"\nAPoT Configuration Summary:")
print(f"- Number of terms: {apot_recipe.num_terms}")
print(f"- Weight bits: {apot_recipe.apot_bits}")
print(f"- Activation bits: 8")
print(f"- Strategy: Channel-wise weights, Tensor-wise activations")