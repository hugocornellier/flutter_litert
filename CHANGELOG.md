## 0.1.3
* Fix Windows: build and bundle custom ops DLL (tflite_custom_ops.dll) for MediaPipe models
* Fix heap corruption crash when switching between segmentation models (custom op name string was freed prematurely)

## 0.1.2
* Fix Linux: build and bundle custom ops library (libtflite_custom_ops.so) so MediaPipe models with custom ops (e.g. selfie segmentation) work on Linux

## 0.1.1
* Update AndroidManifest.xml

## 0.1.0
* Fix IsolateInterpreter thread-safety bug causing intermittent native crashes when hardware delegates are active

## 0.0.1
* Initial release, forked from tflite_flutter_custom v1.2.5
* Rebranded to flutter_litert for LiteRT ecosystem
* All native libraries bundled automatically
* Custom ops support (MediaPipe models)
* Full platform support: Android, iOS, macOS, Windows, Linux
