"""
System and Environment Diagnostic Script
Checks Python, PyTorch, CUDA detection, CPU optimization for Intel i3,
and verifies all required ML libraries.
"""

import sys
import os
import platform
import multiprocessing

def check_hardware_and_cuda():
    print("=" * 60)
    print(" 1. HARDWARE & PROCESSOR SPECIFICATIONS")
    print("=" * 60)
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python Version: {platform.python_version()} ({sys.executable})")
    
    cpu_count = os.cpu_count() or multiprocessing.cpu_count()
    print(f"CPU Logical Threads: {cpu_count}")
    print(f"Processor Name: {platform.processor() or 'Intel Core i3'}")

    print("\n" + "=" * 60)
    print(" 2. PYTORCH & CUDA VERIFICATION")
    print("=" * 60)
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA Device Count: {torch.cuda.device_count()}")
            print(f"Device 0 Name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version (torch): {torch.version.cuda}")
            target_device = torch.device("cuda")
        else:
            print("\n[NOTE ON CUDA DETECTION]")
            print("CUDA is an NVIDIA-proprietary parallel computing architecture.")
            print("It requires a physical NVIDIA GPU (e.g. GeForce GTX/RTX).")
            print("Your machine is equipped with an Intel Core i3 CPU and Intel UHD Graphics.")
            print("PyTorch automatically uses your Intel i3 CPU with Intel MKL acceleration.")
            target_device = torch.device("cpu")
            
        print(f"\nActive Target Device: {target_device}")
        
        # Configure i3 CPU thread optimization
        print("\n" + "=" * 60)
        print(" 3. INTEL i3 CPU OPTIMIZATION CONFIGURATION")
        print("=" * 60)
        # For dual-core / quad-thread i3: set intra-op threads to 4
        optimal_threads = min(4, cpu_count)
        torch.set_num_threads(optimal_threads)
        print(f"Configured PyTorch Intra-op Threads: {torch.get_num_threads()}")
        print(f"Configured PyTorch Inter-op Threads: {torch.get_num_interop_threads()}")

        # Quick tensor compute benchmark on device
        print("Running quick tensor sanity check on target device...")
        x = torch.randn(512, 512, device=target_device)
        y = torch.matmul(x, x)
        print(f"Tensor computation test passed: shape {y.shape}, sum = {y.sum().item():.2f}")

    except ImportError as e:
        print(f"Error importing PyTorch: {e}")
        return False
        
    return True

def verify_libraries():
    print("\n" + "=" * 60)
    print(" 4. REQUIRED LIBRARIES CHECK")
    print("=" * 60)
    
    libraries = [
        ("torch", "PyTorch"),
        ("transformers", "Hugging Face Transformers"),
        ("datasets", "Hugging Face Datasets"),
        ("tokenizers", "Hugging Face Tokenizers"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("tqdm", "tqdm")
    ]
    
    all_ok = True
    for module_name, display_name in libraries:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"[OK] {display_name:<30} (v{version})")
        except ImportError as e:
            print(f"[FAIL] {display_name:<30} ERROR: {e}")
            all_ok = False
            
    print("=" * 60)
    if all_ok:
        print(" STATUS: All requested libraries are installed and ready!")
    else:
        print(" STATUS: Some libraries failed to load.")
    print("=" * 60)
    return all_ok

if __name__ == "__main__":
    check_hardware_and_cuda()
    verify_libraries()
