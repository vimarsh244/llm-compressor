"""Complete workflow script: quantize models and then test them."""

import subprocess
import sys
import time
from pathlib import Path


def run_command(command, description):
    """Run a command and handle output."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print("SUCCESS!")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with return code {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False


def check_model_exists(model_path):
    """Check if a model directory exists."""
    return Path(model_path).exists()


def main():
    print("COMPLETE APoT/PoT QUANTIZATION AND TESTING WORKFLOW")
    print("="*80)
    
    # Model paths
    apot_model_path = "TinyLlama-1.1B-Chat-v1.0-apot-w4a8-t2"
    pot_model_path = "TinyLlama-1.1B-Chat-v1.0-pot-w4a8"
    
    # Check if models already exist
    apot_exists = check_model_exists(apot_model_path)
    pot_exists = check_model_exists(pot_model_path)
    
    print(f"APoT model exists: {apot_exists}")
    print(f"PoT model exists: {pot_exists}")
    
    # Step 1: Create APoT model if it doesn't exist
    if not apot_exists:
        success = run_command(
            "python llama_apot_example.py",
            "Creating APoT quantized model"
        )
        if not success:
            print("Failed to create APoT model. Exiting.")
            sys.exit(1)
        time.sleep(2)
    else:
        print(f"\nSKIPPING: APoT model already exists at {apot_model_path}")
    
    # Step 2: Create PoT model if it doesn't exist
    if not pot_exists:
        success = run_command(
            "python llama_pot_example.py",
            "Creating PoT quantized model"
        )
        if not success:
            print("Failed to create PoT model. Exiting.")
            sys.exit(1)
        time.sleep(2)
    else:
        print(f"\nSKIPPING: PoT model already exists at {pot_model_path}")
    
    # Step 3: Test individual models
    print(f"\n{'='*80}")
    print("TESTING PHASE")
    print(f"{'='*80}")
    
    # Test APoT model
    if check_model_exists(apot_model_path):
        run_command(
            "python test_apot_model.py",
            "Testing APoT model"
        )
        time.sleep(2)
    
    # Test PoT model
    if check_model_exists(pot_model_path):
        run_command(
            "python test_pot_model.py", 
            "Testing PoT model"
        )
        time.sleep(2)
    
    # Step 4: Run combined test
    run_command(
        "python test_quantized_models.py --model-type both",
        "Running combined test for both models"
    )
    time.sleep(2)
    
    # Step 5: Run benchmark comparison
    run_command(
        "python benchmark_quantized_models.py",
        "Running performance benchmark comparison"
    )
    
    print(f"\n{'='*80}")
    print("WORKFLOW COMPLETED!")
    print(f"{'='*80}")
    print(f"APoT model location: {apot_model_path}")
    print(f"PoT model location: {pot_model_path}")
    print("\nYou can now:")
    print("1. Run individual tests: python test_apot_model.py or python test_pot_model.py")
    print("2. Run combined tests: python test_quantized_models.py")
    print("3. Run benchmarks: python benchmark_quantized_models.py")
    print("4. Use the models in your own applications")


if __name__ == "__main__":
    main()
