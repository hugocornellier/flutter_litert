"""Reference outputs for the Play services real-model tests.

Runs each model on the standard LiteRT CPU runtime (with and without
XNNPACK) on a deterministic input that the Dart test rebuilds, prints how far
the two differ, and writes the XNNPACK outputs as little-endian float32 files
named <model>.<output>.f32. Needs the ai-edge-litert package:

    python3 -I scripts/make_real_model_references.py \
      example/play_services_test_host/assets \
      example/play_services_test_host/assets/face_detection_short_range.tflite \
      example/play_services_test_host/assets/mobilefacenet.tflite
"""
import sys
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType

out_dir = Path(sys.argv[1])
models = sys.argv[2:]


def make_input(shape):
    n = int(np.prod(shape))
    i = np.arange(n, dtype=np.int64)
    # Same formula as _patternInput in the Dart test.
    return (((i * 7919) % 256) / 127.5 - 1.0).astype(np.float32).reshape(shape)


def run(path, resolver):
    it = Interpreter(model_path=path, experimental_op_resolver_type=resolver)
    it.allocate_tensors()
    d = it.get_input_details()[0]
    it.set_tensor(d['index'], make_input(d['shape']))
    it.invoke()
    return [(o['name'], it.get_tensor(o['index']).astype(np.float32).ravel())
            for o in it.get_output_details()]


for path in models:
    stem = Path(path).stem
    xnn = run(path, OpResolverType.AUTO)
    ref = run(path, OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    for (name, a), (_, b) in zip(xnn, ref):
        scale = float(np.max(np.abs(a)))
        diff = float(np.max(np.abs(a - b)))
        print(f'{stem} {name}: n={a.size} max|y|={scale:.4g} '
              f'xnnpack-vs-builtin max diff={diff:.3g} ({diff / scale:.3g} of max)')
        safe = name.replace(':', '_').replace('/', '_')
        (out_dir / f'{stem}.{safe}.f32').write_bytes(a.astype('<f4').tobytes())
