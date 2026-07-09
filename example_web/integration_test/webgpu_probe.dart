import 'package:flutter/foundation.dart' show debugPrint;
import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_litert_example_web/detector.dart';
import 'package:integration_test/integration_test.dart';

Future<bool>? _cached;

/// Whether this browser can compile the bundled YOLOv8n model on strict
/// WebGPU (an `{Accelerator.gpu}` request with no WASM fallback).
///
/// This is the capability probe the GPU-flavored tests key off, so a single
/// suite runs everywhere: where WebGPU is present (a workstation browser)
/// they assert real WebGPU compilation and inference, and where it is absent
/// (headless CI runners) they assert the documented fallback or failure
/// semantics instead. Probing by compiling the actual model, rather than
/// sniffing `navigator.gpu`, keeps the probe aligned with exactly what the
/// tests then demand of the runtime.
Future<bool> hasWebGpu() {
  return _cached ??= () async {
    bool available;
    try {
      final probe = await Detector.create(
        config: const WebEngineConfig(
          kind: WebEngineKind.compiledModel,
          accelerators: {Accelerator.gpu},
        ),
      );
      probe.close();
      debugPrint('webgpu probe: available');
      available = true;
    } catch (e) {
      debugPrint('webgpu probe: unavailable ($e)');
      available = false;
    }
    // Drive runs do not relay browser-console prints, so also surface the
    // probed branch in the response data the driver writes to
    // build/integration_response_data.json.
    final binding = IntegrationTestWidgetsFlutterBinding.instance;
    binding.reportData = {...?binding.reportData, 'webgpu': available};
    return available;
  }();
}
