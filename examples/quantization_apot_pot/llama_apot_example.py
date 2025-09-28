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

# this is for if there are multi gpus - ideally will use them
import torch
from llmcompressor.transformers.compression.helpers import calculate_offload_device_map

device_map = calculate_offload_device_map(
    MODEL_ID,
    reserve_for_hessians=True,
    num_gpus=torch.cuda.device_count(),
    trust_remote_code=True,
)

# Select model and load it
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"Loading model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map=device_map)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Select calibration dataset
# DATASET_ID = "mit-han-lab/pile-val-backup"
# DATASET_SPLIT = "validation"

DATASET_ID = "garage-bAInd/Open-Platypus"
DATASET_SPLIT = "train"

# Select number of samples for calibration
NUM_CALIBRATION_SAMPLES = 20480
MAX_SEQUENCE_LENGTH = 2048

print(f"Loading calibration dataset: {DATASET_ID}")
# ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
# ds = ds.shuffle(seed=42)


# def preprocess(example):
#     """Preprocess the dataset examples."""
#     return {
#         "text": tokenizer.apply_chat_template(
#             [{"role": "user", "content": example["text"]}],
#             tokenize=False,
#         )
#     }


# ds = ds.map(preprocess)


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
    # config_groups={
    #     "group_0": QuantizationScheme(
    #         targets=["Linear"],
    #         weights=QuantizationArgs(
    #             num_bits=8,
    #             type="int",
    #             symmetric=True,
    #             strategy="channel",  # per-channel quantization for better accuracy
    #             observer="apot",
    #             observer_kwargs={"num_terms": 4},
    #         ),
    #         input_activations=QuantizationArgs(
    #             num_bits=8,
    #             type="int",
    #             symmetric=True,
    #             strategy="tensor",  # per-tensor quantization for activations
    #             observer="apot",
    #             observer_kwargs={"num_terms": 4},
    #         )
    #     )
    # },
    ignore=["lm_head"],  # typically keep the output layer at full precision
    apot_bits=8,  # use 8-bit APoT quantization for weights
    num_terms=4,  # use 4-term APoT (sum of 4 signed powers of two)
)

print(f"Applying APoT quantization with {apot_recipe.num_terms} terms...")

# Apply APoT quantization
oneshot(
    model=model,
    # dataset=ds,
    dataset="open_platypus",
    recipe=apot_recipe,
    output_dir="TinyLlama-1.1B-Chat-v1.0-apot-w8a8-t4",
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("APoT quantization completed!")

# Test the quantized model
print("\n" + "="*50)
print("TESTING QUANTIZED MODEL GENERATION")
print("="*50)


from vllm import LLM
model = LLM("TinyLlama-1.1B-Chat-v1.0-apot-w8a8-t4", device_map=device_map)
output = model.generate("The python code to generate first 1000 digits of pi is:")
