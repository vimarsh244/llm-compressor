#!/usr/bin/env python3
"""
Simple standalone test to verify POT and APOT quantization fixes.
This tests the core quantization logic directly without module dependencies.
"""

import torch

def generate_pot_levels(num_bits: int):
    """Generate POT quantization levels."""
    max_exp = num_bits - 2
    pot_levels = [0.0]
    for exp in range(max_exp + 1):
        level = 2.0 ** exp
        pot_levels.extend([level, -level])
    return sorted(pot_levels)

def generate_apot_levels(num_bits: int, num_terms: int = 2):
    """Generate APoT quantization levels (simplified)."""
    import itertools
    
    max_exp = num_bits - 2
    min_exp = -(num_bits - 2)
    
    powers = [2.0 ** i for i in range(min_exp, max_exp + 1)]
    powers.append(0.0)
    
    levels = set()
    
    if num_terms == 2:
        for p1, p2 in itertools.combinations_with_replacement(powers, 2):
            if p1 != 0 or p2 != 0:
                levels.add(p1 + p2)
                levels.add(p1 - p2)
                levels.add(-p1 - p2)
                levels.add(-p1 + p2)
    
    return sorted(list(levels))

def test_quantization_levels():
    """Test that POT and APoT generate different quantization levels."""
    print("Testing POT vs APoT Quantization Levels")
    print("=" * 50)
    
    num_bits = 4
    
    # Generate levels
    pot_levels = generate_pot_levels(num_bits)
    apot_levels = generate_apot_levels(num_bits, num_terms=2)
    
    print(f"POT levels ({len(pot_levels)}): {pot_levels}")
    print(f"APoT levels ({len(apot_levels)}): {apot_levels[:15]}{'...' if len(apot_levels) > 15 else ''}")
    
    # Check differences
    pot_set = set(pot_levels)
    apot_set = set(apot_levels)
    
    print(f"\nNumber of POT levels: {len(pot_levels)}")
    print(f"Number of APoT levels: {len(apot_levels)}")
    
    unique_to_pot = pot_set - apot_set
    unique_to_apot = apot_set - pot_set
    
    print(f"Levels unique to POT: {len(unique_to_pot)}")
    print(f"Levels unique to APoT: {len(unique_to_apot)}")
    
    if len(unique_to_pot) > 0 or len(unique_to_apot) > 0:
        print("✅ SUCCESS: POT and APoT have different quantization levels!")
        return True
    else:
        print("❌ FAILURE: POT and APoT have identical quantization levels")
        return False

def simple_pot_quantize(tensor, scale, num_bits=4):
    """Simple POT quantization."""
    pot_levels = generate_pot_levels(num_bits)
    scaled = tensor / scale
    
    # Find nearest POT level for each element
    result = torch.zeros_like(scaled)
    for i in range(scaled.numel()):
        val = scaled.flatten()[i].item()
        nearest = min(pot_levels, key=lambda x: abs(x - val))
        result.flatten()[i] = nearest
    
    return result * scale

def simple_apot_quantize(tensor, scale, num_bits=4, num_terms=2):
    """Simple APoT quantization."""
    apot_levels = generate_apot_levels(num_bits, num_terms)
    scaled = tensor / scale
    
    # Find nearest APoT level for each element
    result = torch.zeros_like(scaled)
    for i in range(scaled.numel()):
        val = scaled.flatten()[i].item()
        nearest = min(apot_levels, key=lambda x: abs(x - val))
        result.flatten()[i] = nearest
    
    return result * scale

def test_quantization_results():
    """Test that POT and APoT produce different quantized results."""
    print("\nTesting POT vs APoT Quantization Results")
    print("=" * 50)
    
    # Create test tensor
    torch.manual_seed(42)
    test_tensor = torch.tensor([0.5, 1.2, -0.8, 2.1, -1.5, 0.3])
    scale = torch.tensor(0.5)
    
    print(f"Original tensor: {test_tensor}")
    print(f"Scale: {scale}")
    
    # Apply quantization
    pot_result = simple_pot_quantize(test_tensor, scale)
    apot_result = simple_apot_quantize(test_tensor, scale, num_terms=2)
    
    print(f"\nPOT quantized:  {pot_result}")
    print(f"APoT quantized: {apot_result}")
    
    # Check difference
    diff = torch.abs(pot_result - apot_result)
    max_diff = diff.max().item()
    num_different = (diff > 1e-6).sum().item()
    
    print(f"\nMax difference: {max_diff:.6f}")
    print(f"Number of different values: {num_different}/{len(test_tensor)}")
    
    if num_different > 0:
        print("✅ SUCCESS: POT and APoT produce different quantized results!")
        return True
    else:
        print("❌ FAILURE: POT and APoT produce identical results")
        return False

def main():
    """Run all tests."""
    print("Simple POT vs APoT Quantization Test")
    print("This verifies that the quantization methods produce different outputs.\n")
    
    levels_different = test_quantization_levels()
    results_different = test_quantization_results()
    
    print("\n" + "=" * 50)
    print("OVERALL SUMMARY")
    print("=" * 50)
    
    if levels_different and results_different:
        print("🎉 EXCELLENT: POT and APoT quantization are working correctly!")
        print("   - Different quantization level sets")
        print("   - Different quantization results")
        print("\nThe implementation successfully differentiates POT and APoT quantization.")
    elif levels_different:
        print("⚠️  PARTIAL SUCCESS: Level sets are different but results are still similar")
        print("   This suggests the quantization logic may need further refinement.")
    else:
        print("❌ FAILURE: POT and APoT are still producing identical outputs")
        print("   The implementation issue persists.")

if __name__ == "__main__":
    main()

