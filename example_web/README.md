# flutter_litert_example_web

Web example app for `flutter_litert`, mirroring the native example's
object-detection demo (and the `example/` plus `example_web/` layout used
by the detection packages). Runs real YOLOv8n object detection in the
browser with the same UX as the native app: a sample-image picker, an
inference-engine settings dialog, per-box colored overlays, and a
microsecond timing readout.

The engine dialog covers all three web inference stacks:

- `CompiledModel` (LiteRT.js): WASM, WebGPU + WASM fallback, or WebGPU
  only, with fp16/fp32 precision on WebGPU
- `LiteRtInterpreter` (LiteRT.js): WASM or WebGPU with automatic fallback
- `Interpreter` (tflite-js): WASM

It doubles as the web integration-test host: CI drives this app headlessly
with chromedriver and asserts the cats in `assets/cat.jpg` are detected on
every engine. The model and sample images are symlinks into
`example/assets/`.

## Run the demo app

```sh
flutter run -d chrome
```

## Run the integration tests

```sh
chromedriver --port=4444 &
flutter drive \
  --driver=test_driver/integration_test.dart \
  --target=integration_test/web_object_detection_test.dart \
  -d web-server --browser-name=chrome
```
