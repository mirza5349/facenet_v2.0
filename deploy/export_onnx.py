import os
import torch
from facenet_mirza import InceptionResnetV1, MTCNN

def export_resnet(output_path="weights/facenet_resnet.onnx"):
    print("=== Exporting InceptionResnetV1 (FaceNet) to ONNX ===")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load pretrained model
    model = InceptionResnetV1(pretrained='vggface2').eval()
    
    # Create dummy input representing a single cropped face image (batch_size=1, channels=3, H=160, W=160)
    dummy_input = torch.randn(1, 3, 160, 160, requires_grad=False)
    
    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,  # High-compatibility opset version
        do_constant_folding=True,
        input_names=['input_face'],
        output_names=['face_embedding'],
        dynamic_axes={
            'input_face': {0: 'batch_size'},
            'face_embedding': {0: 'batch_size'}
        }
    )
    print(f"Successfully exported InceptionResnetV1 to: {output_path}")

def export_mtcnn_stages(output_dir="weights/mtcnn"):
    print("=== Exporting MTCNN sub-networks (PNet, RNet, ONet) to ONNX ===")
    os.makedirs(output_dir, exist_ok=True)
    
    mtcnn = MTCNN()
    
    # 1. PNet Export (fully convolutional, takes variable height/width inputs)
    pnet_dummy = torch.randn(1, 3, 12, 12)
    pnet_path = os.path.join(output_dir, "pnet.onnx")
    torch.onnx.export(
        mtcnn.pnet.eval(),
        pnet_dummy,
        pnet_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_pnet'],
        output_names=['pnet_bounding_box', 'pnet_probability'],
        dynamic_axes={
            'input_pnet': {0: 'batch_size', 2: 'height', 3: 'width'},
            'pnet_bounding_box': {0: 'batch_size', 2: 'height', 3: 'width'},
            'pnet_probability': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    print(f"Successfully exported PNet to: {pnet_path}")
    
    # 2. RNet Export
    rnet_dummy = torch.randn(1, 3, 24, 24)
    rnet_path = os.path.join(output_dir, "rnet.onnx")
    torch.onnx.export(
        mtcnn.rnet.eval(),
        rnet_dummy,
        rnet_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_rnet'],
        output_names=['rnet_bounding_box', 'rnet_probability'],
        dynamic_axes={
            'input_rnet': {0: 'batch_size'},
            'rnet_bounding_box': {0: 'batch_size'},
            'rnet_probability': {0: 'batch_size'}
        }
    )
    print(f"Successfully exported RNet to: {rnet_path}")
    
    # 3. ONet Export
    onet_dummy = torch.randn(1, 3, 48, 48)
    onet_path = os.path.join(output_dir, "onet.onnx")
    torch.onnx.export(
        mtcnn.onet.eval(),
        onet_dummy,
        onet_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_onet'],
        output_names=['onet_bounding_box', 'onet_landmarks', 'onet_probability'],
        dynamic_axes={
            'input_onet': {0: 'batch_size'},
            'onet_bounding_box': {0: 'batch_size'},
            'onet_landmarks': {0: 'batch_size'},
            'onet_probability': {0: 'batch_size'}
        }
    )
    print(f"Successfully exported ONet to: {onet_path}")

if __name__ == "__main__":
    export_resnet("weights/facenet_resnet.onnx")
    export_mtcnn_stages("weights/mtcnn")
