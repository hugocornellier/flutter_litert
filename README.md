## flutter_litert

A drop-in LiteRT (formerly TensorFlow Lite) plugin for Flutter, with all native libraries bundled automatically for Android, iOS, macOS, Windows, and Linux.

No manual setup or handling of native libraries required.

### Why this package?

Google's official `litert_flutter` package is still a placeholder. This package provides a production-ready LiteRT integration based on [`tflite_flutter`](https://pub.dev/packages/tflite_flutter), with the friction removed.

* **No more copying `.so`, `.dll`, or `.dylib` files**
* **Works out of the box** on all supported platforms
* **Same API** as `tflite_flutter`
* **Custom ops support** (e.g., MediaPipe models)

### Installation

```yaml
dependencies:
  flutter_litert: ^0.0.1
````

### Usage

```dart
import 'package:flutter_litert/flutter_litert.dart';

void main() async {
  final interpreter = await Interpreter.fromAsset('model.tflite');
  print('Model loaded successfully!');
}
```

### Platform Support

* Android
* iOS
* macOS
* Windows
* Linux

All required native binaries are automatically included in the build.

### Credits

This project is based on [`tflite_flutter`](https://pub.dev/packages/tflite_flutter) by the TensorFlow team and contributors, and [`tflite_flutter_custom`](https://pub.dev/packages/tflite_flutter_custom) by Hugo Cornellier.
