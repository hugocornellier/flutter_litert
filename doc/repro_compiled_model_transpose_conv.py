#!/usr/bin/env python3
"""Minimal repro: LiteRT Next CompiledModel miscomputes TRANSPOSE_CONV on CPU.

This uses ONLY Google's own bindings from the `ai-edge-litert` wheel. flutter_litert
is not involved, which is the point: the same LiteRT build is driven through both
the classic TFLite Interpreter and LiteRT Next CompiledModel, and only the latter is
wrong.

    pip install ai-edge-litert==2.1.6 tensorflow numpy
    python repro_compiled_model_transpose_conv.py

Observed on macOS 15 / arm64 (M4 Max), ai-edge-litert 2.1.6:

    graph                       Interpreter          CompiledModel(cpu)
    conv                        ok                   ok
    deconv_x1                   ok                   WRONG
    deconv_x2                   ok                   raises code=3
    deconv_x2_then_1x1_conv     ok                   ok

The last two rows are the same two Conv2DTranspose layers; the only difference is
whether a trivial 1x1 convolution consumes the deconv output instead of the deconv
writing straight into CompiledModel's caller-supplied output buffer.

The failing rows are not stable across processes: `deconv_x1` has been seen wrong by 54%
of the output range in one process and NaN in another, and `deconv_x2` has raised on run 1
in one process and run 2 in another. The reason is that CompiledModel **does not write the
output buffer at all** on these graphs while still returning `kLiteRtStatusOk`, so what you
observe is whatever the freshly allocated buffer happened to contain. Prefill the output
with a sentinel and it survives the call untouched. Treat the specific magnitudes as
incidental; the assertion is Interpreter/CompiledModel disagreement.

The real trigger is a **dynamic (runtime-shaped) model output**, not TRANSPOSE_CONV as
such. Keras emits Conv2DTranspose output shapes as a runtime PACK, and TFLite then marks
that tensor kTfLiteDynamic; LiteRT installs the caller's buffer as a custom allocation,
preparation flips the tensor to dynamic, and the dynamic allocation is never copied back.
A following 1x1 conv fixes it only by demoting the dynamic tensor to an internal one.

Upstream PR, open and unmerged as of 2026-07-29, not in any release:
https://github.com/google-ai-edge/LiteRT/pull/8667

Expected: CompiledModel matches the Interpreter to float32 tolerance in every row.
"""

import pathlib
import sys
import tempfile

import ai_edge_litert
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(pathlib.Path(ai_edge_litert.__file__).parent))
import _pywrap_litert_compiled_model_wrapper as cm_api  # noqa: E402
import _pywrap_litert_environment_wrapper as env_api  # noqa: E402
import _pywrap_litert_tensor_buffer_wrapper as tb_api  # noqa: E402

KLITERT_HW_CPU = 1
L = tf.keras.layers


def _build_graphs(outdir):
    """Return {name: tflite_path}. Float32, no quantization."""
    graphs = {}

    def emit(name, model):
        path = outdir / f"{name}.tflite"
        path.write_bytes(tf.lite.TFLiteConverter.from_keras_model(model).convert())
        graphs[name] = path

    inp = tf.keras.Input((12, 12, 128))
    emit("conv", tf.keras.Model(inp, L.Conv2D(64, 3, padding="same")(inp)))

    inp = tf.keras.Input((12, 12, 128))
    emit(
        "deconv_x1",
        tf.keras.Model(inp, L.Conv2DTranspose(64, 4, strides=2, padding="same")(inp)),
    )

    inp = tf.keras.Input((12, 12, 128))
    x = L.Conv2DTranspose(64, 4, strides=2, padding="same")(inp)
    x = L.Conv2DTranspose(32, 4, strides=2, padding="same")(x)
    emit("deconv_x2", tf.keras.Model(inp, x))

    inp = tf.keras.Input((12, 12, 128))
    x = L.Conv2DTranspose(64, 4, strides=2, padding="same")(inp)
    x = L.Conv2DTranspose(32, 4, strides=2, padding="same")(x)
    x = L.Conv2D(32, 1, padding="same")(x)
    emit("deconv_x2_then_1x1_conv", tf.keras.Model(inp, x))

    return graphs


def _reference(path, x):
    """Ground truth via the classic TFLite interpreter."""
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    interp.set_tensor(interp.get_input_details()[0]["index"], x)
    interp.invoke()
    return interp.get_tensor(interp.get_output_details()[0]["index"]).copy()


def _compiled_model(env, path, x, expected, runs=3):
    """Run the same graph through CompiledModel. Returns a printable verdict."""
    try:
        model = cm_api.CreateCompiledModelFromFile(env, str(path), KLITERT_HW_CPU)
        inputs = model.CreateInputBuffers(0)
        outputs = model.CreateOutputBuffers(0)
    except Exception as exc:  # noqa: BLE001
        return f"construction failed: {str(exc).splitlines()[0][:60]}"

    deviations = []
    for run in range(runs):
        try:
            tb_api.WriteTensor(inputs[0], x, "float32")
            model.Run(inputs, outputs)
            got = np.asarray(
                tb_api.ReadTensor(outputs[0], expected.size, "float32")
            ).reshape(expected.shape)
        except Exception as exc:  # noqa: BLE001
            return f"run {run + 1} raised: {str(exc).splitlines()[0][:52]}"
        deviations.append(float(np.abs(got - expected).max()))

    spread = float(expected.max() - expected.min())
    pct = 100.0 * deviations[0] / spread if spread else 0.0
    verdict = "ok" if pct < 1.0 else "WRONG"
    return f"{verdict}, dev {deviations[0]:.2e} ({pct:.1f}% of range), {runs}/{runs} runs"


def main():
    print(f"ai-edge-litert {ai_edge_litert.__version__}, tensorflow {tf.__version__}")
    env = env_api.CreateEnvironment()

    with tempfile.TemporaryDirectory() as tmp:
        graphs = _build_graphs(pathlib.Path(tmp))
        print(f"\n{'graph':26s} {'Interpreter':>14s}  CompiledModel(cpu)")
        failures = 0

        for name, path in graphs.items():
            interp = tf.lite.Interpreter(model_path=str(path))
            interp.allocate_tensors()
            shape = interp.get_input_details()[0]["shape"]
            x = np.random.default_rng(31).random(shape, dtype=np.float32)

            expected = _reference(path, x)
            verdict = _compiled_model(env, path, x, expected)
            if not verdict.startswith("ok"):
                failures += 1
            print(f"{name:26s} {'ok':>14s}  {verdict}")

    print(
        f"\n{failures} of {len(graphs)} graphs disagree with the Interpreter "
        "under CompiledModel."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
