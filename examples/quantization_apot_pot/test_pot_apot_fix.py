#!/usr/bin/env python3
"""
Test script to verify that POT and APOT quantization now produce different results.
This script tests the observers and quantization utilities directly.
"""

import torch
import sys
import os
import numpy as np

# Add the source directory to Python path
sys.path.insert(0, '/home/vimarsh/Desktop/4-1/RC/llm-compressor/src')

from llmcompressor.observers.pot_apot import PoTObserver, APoTObserver
from llmcompressor.modifiers.quantization.pot.utils import fake_quantize_pot
from llmcompressor.modifiers.quantization.apot.utils import fake_quantize_apot, generate_apot_levels
from compressed_tensors.quantization import QuantizationArgs

def test_observers():
    """Test that POT and APOT observers produce different scales and behaviors."""
    print("=" * 60)
    print("TESTING POT vs APOT OBSERVERS")
    print("=" * 60)
    
    # Create test tensor
    torch.manual_seed(42)
    test_tensor = torch.randn(4, 4) * 2.0
    print(f"Test tensor:\n{test_tensor}")
    print(f"Max abs value: {test_tensor.abs().max():.4f}")
    
    # Create quantization args
    quant_args = QuantizationArgs(
        num_bits=4,
        type="int",
        symmetric=True,
        strategy="tensor"
    )
    
    # Test POT observer
    pot_observer = PoTObserver(quant_args)
    pot_scale, pot_zp = pot_observer.calculate_qparams(test_tensor)
    print(f"\nPOT Observer:")
    print(f"  Scale: {pot_scale}")
    print(f"  Zero Point: {pot_zp}")
    
    # Test APOT observer with 2 terms
    apot_observer = APoTObserver(quant_args, num_terms=2)
    apot_scale, apot_zp = apot_observer.calculate_qparams(test_tensor)
    print(f"\nAPOT Observer (2 terms):")
    print(f"  Scale: {apot_scale}")
    print(f"  Zero Point: {apot_zp}")
    
    # Check if scales are different
    scale_diff = abs(pot_scale - apot_scale).item()
    print(f"\nScale difference: {scale_diff:.6f}")
    
    if scale_diff < 1e-6:
        print("⚠️  WARNING: Scales are too similar!")
    else:
        print("✅ Good: Scales are different")

def test_quantization_levels():
    """Test the actual quantization levels produced by POT vs APOT."""
    print("\n" + "=" * 60)
    print("TESTING QUANTIZATION LEVELS")
    print("=" * 60)
    
    num_bits = 4
    
    # Generate POT levels manually
    max_exp = num_bits - 2
    pot_levels = [0.0]
    for exp in range(max_exp + 1):
        level = 2.0 ** exp
        pot_levels.extend([level, -level])
    pot_levels = sorted(pot_levels)
    
    # Generate APOT levels
    apot_levels = generate_apot_levels(num_bits, num_terms=2)
    
    print(f"POT levels ({len(pot_levels)}): {pot_levels}")
    print(f"APOT levels ({len(apot_levels)}): {apot_levels[:10]}{'...' if len(apot_levels) > 10 else ''}")
    print(f"Number of POT levels: {len(pot_levels)}")
    print(f"Number of APOT levels: {len(apot_levels)}")
    
    # Check if level sets are different
    pot_set = set(pot_levels)
    apot_set = set(apot_levels)
    
    if pot_set == apot_set:
        print("⚠️  WARNING: POT and APOT levels are identical!")
    else:
        print("✅ Good: POT and APOT levels are different")
        print(f"   Unique to POT: {len(pot_set - apot_set)} levels")
        print(f"   Unique to APOT: {len(apot_set - pot_set)} levels")

def test_fake_quantization():
    """Test that fake quantization produces different results."""
    print("\n" + "=" * 60)
    print("TESTING FAKE QUANTIZATION")
    print("=" * 60)
    
    # Create test tensor
    torch.manual_seed(42)
    test_tensor = torch.randn(2, 4) * 1.5
    print(f"Original tensor:\n{test_tensor}")
    
    # Common parameters
    scale = torch.tensor(0.25)
    zero_point = torch.tensor(0)
    num_bits = 4
    
    # Apply POT fake quantization
    pot_result = fake_quantize_pot(test_tensor, scale, zero_point, num_bits)
    print(f"\nPOT quantized:\n{pot_result}")
    
    # Apply APOT fake quantization
    apot_result = fake_quantize_apot(test_tensor, scale, zero_point, num_bits, num_terms=2)
    print(f"\nAPOT quantized:\n{apot_result}")
    
    # Check if results are different
    diff = torch.abs(pot_result - apot_result).max().item()
    print(f"\nMax difference: {diff:.6f}")
    
    if diff < 1e-6:
        print("⚠️  WARNING: POT and APOT results are too similar!")
        return False
    else:
        print("✅ Good: POT and APOT produce different results")
        return True

def test_model_inference():
    """Test that POT and APOT observers work in model context."""
    print("\n" + "=" * 60)
    print("TESTING MODEL OBSERVER INTEGRATION")
    print("=" * 60)
    
    # Create test tensor
    torch.manual_seed(42)
    test_tensor = torch.randn(2, 4) * 2.0
    
    # Create quantization args
    quant_args = QuantizationArgs(
        num_bits=4,
        type="int",
        symmetric=True,
        strategy="tensor"
    )
    
    # Test POT observer with fake quantization
    pot_observer = PoTObserver(quant_args)
    pot_fake_quant = pot_observer.fake_quantize(test_tensor)
    
    # Test APOT observer with fake quantization
    apot_observer = APoTObserver(quant_args, num_terms=2)
    apot_fake_quant = apot_observer.fake_quantize(test_tensor)
    
    print(f"Original:\n{test_tensor}")
    print(f"\nPOT fake quantized:\n{pot_fake_quant}")
    print(f"\nAPOT fake quantized:\n{apot_fake_quant}")
    
    # Check difference
    diff = torch.abs(pot_fake_quant - apot_fake_quant).max().item()
    print(f"\nMax difference: {diff:.6f}")
    
    if diff < 1e-6:
        print("⚠️  WARNING: Observer fake quantization results are too similar!")
        return False
    else:
        print("✅ Good: Observer fake quantization produces different results")
        return True

def main():
    """Run all tests."""
    print("Testing POT vs APOT Quantization Fix")
    print("This script verifies that the implemented fix makes POT and APOT produce different results.")
    
    try:
        # Run all tests
        test_observers()
        test_quantization_levels()
        quant_different = test_fake_quantization()
        observer_different = test_model_inference()
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        if quant_different and observer_different:
            print("✅ SUCCESS: POT and APOT quantization are now working correctly!")
            print("   - Different quantization levels")
            print("   - Different fake quantization results")
            print("   - Different observer behaviors")
            print("\nThe fix has resolved the issue where POT and APOT were producing identical outputs.")
        else:
            print("❌ ISSUES REMAIN: Some tests show POT and APOT are still too similar.")
            print("   Additional debugging may be needed.")
            
    except Exception as e:
        print(f"\n❌ ERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
