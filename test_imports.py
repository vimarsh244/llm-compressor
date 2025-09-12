#!/usr/bin/env python3
"""
Simple test to check if our new modules can be imported correctly.
"""

def test_imports():
    """Test that all new modules can be imported."""
    try:
        print("Testing PoT observer import...")
        from llmcompressor.observers.pot_apot import PoTObserver
        print("✓ PoT observer imported successfully")
        
        print("Testing APoT observer import...")
        from llmcompressor.observers.pot_apot import APoTObserver
        print("✓ APoT observer imported successfully")
        
        print("Testing PoT modifier import...")
        from llmcompressor.modifiers.quantization.pot import PoTQuantizationModifier
        print("✓ PoT modifier imported successfully")
        
        print("Testing APoT modifier import...")
        from llmcompressor.modifiers.quantization.apot import APoTQuantizationModifier
        print("✓ APoT modifier imported successfully")
        
        print("Testing PoT utils import...")
        from llmcompressor.modifiers.quantization.pot.utils import quantize_pot
        print("✓ PoT utils imported successfully")
        
        print("Testing APoT utils import...")
        from llmcompressor.modifiers.quantization.apot.utils import generate_apot_levels
        print("✓ APoT utils imported successfully")
        
        print("\n🎉 All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality of the new classes."""
    try:
        print("\nTesting basic functionality...")
        
        from compressed_tensors.quantization.quant_args import QuantizationArgs
        from llmcompressor.observers.pot_apot import PoTObserver, APoTObserver
        
        # test PoT observer
        quant_args = QuantizationArgs(num_bits=4, symmetric=True, type="int")
        pot_observer = PoTObserver(quant_args)
        print("✓ PoT observer created")
        
        # test APoT observer
        apot_observer = APoTObserver(quant_args, num_terms=2)
        print("✓ APoT observer created")
        
        # test with a simple tensor
        import torch
        tensor = torch.tensor([1.0, 2.0, 4.0])
        
        scale, zp = pot_observer.calculate_qparams(tensor)
        print(f"✓ PoT qparams calculated: scale={scale}, zp={zp}")
        
        scale, zp = apot_observer.calculate_qparams(tensor)
        print(f"✓ APoT qparams calculated: scale={scale}, zp={zp}")
        
        print("\n🎉 Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*50)
    print("Testing PoT/APoT Implementation")
    print("="*50)
    
    imports_ok = test_imports()
    if imports_ok:
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n✅ All tests passed! Implementation is ready.")
        else:
            print("\n❌ Functionality tests failed.")
    else:
        print("\n❌ Import tests failed.")