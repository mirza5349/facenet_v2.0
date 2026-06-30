import os
import sys

def build_engine(onnx_path="weights/facenet_resnet.onnx", engine_path="weights/facenet_resnet.engine", use_fp16=True):
    print(f"=== Compiling ONNX model {onnx_path} to TensorRT Engine ===")
    
    # 1. Try to use programmatic Python TensorRT bindings
    try:
        import tensorrt as trt
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        
        # Use EXPLICIT_BATCH flag for standard ONNX parser integration
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # Read ONNX file
        if not os.path.exists(onnx_path):
            print(f"Error: ONNX model {onnx_path} not found. Please run export_onnx.py first!")
            return False
            
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('Error: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return False
                
        config = builder.create_builder_config()
        # Set max workspace size (e.g., 1 GB)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        
        # Apply FP16 optimizations if requested and supported
        if use_fp16:
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("Applied FP16 Precision optimization flag.")
            else:
                print("Warning: Platform does not support fast FP16. Falling back to FP32.")
                
        # Build the serialized network
        print("Building TensorRT engine... This may take a few minutes...")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("Error: Failed to build the engine.")
            return False
            
        # Write to disk
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
            
        print(f"Successfully compiled and saved TensorRT engine to: {engine_path}")
        return True
        
    except ImportError:
        # 2. Fall back to trtexec shell instructions (standard on Jetson JetPack)
        print("\n[Notice] 'tensorrt' Python module not detected. This is normal on standard CPU/cloud environments.")
        print("NVIDIA Jetson devices typically pre-install TensorRT and include the 'trtexec' CLI utility.")
        print("\nTo compile the engine directly on your NVIDIA Jetson, run the following CLI command:")
        
        fp16_flag = "--fp16" if use_fp16 else ""
        trtexec_cmd = f"trtexec --onnx={onnx_path} --saveEngine={engine_path} {fp16_flag} --workspace=1024"
        
        print("-" * 80)
        print(f"  {trtexec_cmd}")
        print("-" * 80)
        return False

if __name__ == "__main__":
    build_engine()
