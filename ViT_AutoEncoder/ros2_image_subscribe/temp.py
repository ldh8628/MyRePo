import onnxruntime as ort
import numpy as np

onnx_path = "yolo_nas_s.onnx"

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
print("input name:", inp.name, "shape:", inp.shape, "dtype:", inp.type)

dummy = np.zeros((1, 3, 640, 640), dtype=np.uint8)
outs = sess.run(None, {inp.name: dummy})

for i, o in enumerate(outs):
    print(f"output[{i}] shape={o.shape}, dtype={o.dtype}")
