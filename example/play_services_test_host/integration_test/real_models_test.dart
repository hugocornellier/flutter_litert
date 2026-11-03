import 'package:flutter/foundation.dart';
import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'real_models.dart';

/// The real-model checks on their own, to compare runtimes on one device.
///
/// play_services_runtime_test.dart runs the same checks. This file also works
/// when the host is built with the bundled runtime, which an environment
/// variable selects because Gradle properties cannot be passed to flutter test:
///
/// ```bash
/// env 'ORG_GRADLE_PROJECT_flutterLitert.runtime=bundled' \
///   flutter test integration_test/real_models_test.dart -d <device>
/// ```
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late LiteRtRuntimeStatus status;

  setUpAll(() async {
    status = await initializeAndroidRuntime(enableGpu: true);
    debugPrint('flutter_litert runtime: $androidRuntime, status: $status');
  });

  defineRealModelTests(() => status);
}
