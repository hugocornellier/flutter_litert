import 'package:flutter/material.dart';
import 'package:flutter_litert/flutter_litert.dart' show Precision;
import 'package:flutter_litert_example_web/main.dart' as app;
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'webgpu_probe.dart';

/// Drives the real example_web app end to end in a browser: boot-time
/// compile and detect, the sample-image picker, and every path through the
/// inference-engine settings dialog. Engine correctness is covered by
/// web_object_detection_test.dart; these tests assert the app around the
/// engines: the status readout, the detection list, the overlay, and
/// settings round-trips (apply, cancel, and recovery from an engine that
/// fails to compile).
///
/// The suite never uses pumpAndSettle: the app shows an indeterminate
/// spinner until the first result lands (and again after an engine error),
/// which would spin forever. Everything polls with [_pumpUntil] instead.
///
/// Run with:
///   chromedriver --port=4444 &
///   flutter drive --profile --driver=test_driver/integration_test.dart \
///     --target=integration_test/web_app_ui_test.dart \
///     -d web-server --browser-name=chrome
void main() {
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  // Every engine switch recompiles a model; the first compile in the session
  // also fetches the runtime and its WASM binary from a CDN.
  const timeout = Timeout(Duration(minutes: 10));

  // Matches the idle status line, e.g. 'WASM · 48213µs · 4 detections'.
  final statusRe = RegExp(
    r'^(WebGPU|WASM|tflite-js \(WASM\)) · \d+µs · (\d+) detections$',
  );

  String? textWhere(WidgetTester tester, bool Function(String) test) {
    for (final w in tester.widgetList<Text>(find.byType(Text))) {
      final s = w.data;
      if (s != null && test(s)) return s;
    }
    return null;
  }

  String? statusLine(WidgetTester tester) =>
      textWhere(tester, statusRe.hasMatch);

  String? errorLine(WidgetTester tester) =>
      textWhere(tester, (s) => s.startsWith('Engine failed:'));

  String backendOf(String status) => statusRe.firstMatch(status)!.group(1)!;

  int detectionCount(String status) =>
      int.parse(statusRe.firstMatch(status)!.group(2)!);

  Future<void> pumpUntil(
    WidgetTester tester,
    bool Function() condition,
    String what, {
    Duration limit = const Duration(minutes: 2),
  }) async {
    final deadline = DateTime.now().add(limit);
    while (!condition()) {
      if (DateTime.now().isAfter(deadline)) {
        // Web drive runs report failure details as an empty string, so also
        // record what was being waited for, and what the app showed, in the
        // response data the driver writes to
        // build/integration_response_data.json.
        final visible = tester
            .widgetList<Text>(find.byType(Text))
            .map((w) => w.data)
            .whereType<String>()
            .take(30)
            .toList();
        binding.reportData = {
          ...?binding.reportData,
          'timedOutWaitingFor': what,
          'visibleText': visible,
        };
        fail('timed out waiting for $what');
      }
      await tester.pump(const Duration(milliseconds: 200));
    }
  }

  /// Pumps until a detection cycle that started after [previous] completes,
  /// in either a fresh status line (returned) or an engine error (null).
  Future<String?> waitForIdle(WidgetTester tester, {String? previous}) async {
    String? status;
    await pumpUntil(tester, () {
      if (errorLine(tester) != null) return true;
      status = statusLine(tester);
      return status != null && status != previous;
    }, 'a detection result (previous: $previous)');
    return status;
  }

  /// Pumps the real app and waits for the default engine's first detection.
  Future<String> bootApp(WidgetTester tester) async {
    await tester.pumpWidget(const app.WebExampleApp());
    final status = await waitForIdle(tester);
    expect(
      status,
      isNotNull,
      reason: 'default engine must boot; got: ${errorLine(tester)}',
    );
    return status!;
  }

  Future<void> openSettings(WidgetTester tester) async {
    await tester.tap(find.byIcon(Icons.tune));
    await pumpUntil(
      tester,
      () => find.text('Apply').evaluate().isNotEmpty,
      'the settings dialog',
      limit: const Duration(seconds: 30),
    );
    // Let the dialog's fade/scale transition finish before hit-testing.
    await tester.pump(const Duration(milliseconds: 350));
  }

  Future<void> tapInDialog(WidgetTester tester, String label) async {
    final finder = find.text(label);
    await tester.ensureVisible(finder);
    await tester.pump();
    await tester.tap(finder);
    await tester.pump();
  }

  /// Applies the dialog and waits for the resulting recompile+detect cycle.
  Future<String?> applyAndWait(WidgetTester tester, String? previous) async {
    await tester.tap(find.text('Apply'));
    await tester.pump();
    return waitForIdle(tester, previous: previous);
  }

  /// Sets the confidence slider to roughly [fraction] of its range by
  /// tapping the track (Material sliders jump to the tapped position and
  /// snap to the nearest division), then waits for the re-detection that
  /// onChangeEnd kicks off.
  Future<String?> setConfidence(
    WidgetTester tester,
    double fraction,
    String? previous,
  ) async {
    final rect = tester.getRect(find.byType(Slider));
    const trackInset = 24.0; // Material slider track padding per side.
    final x = rect.left + trackInset + (rect.width - 2 * trackInset) * fraction;
    await tester.tapAt(Offset(x, rect.center.dy));
    await tester.pump();
    return waitForIdle(tester, previous: previous);
  }

  /// Asserts the detection list contains a [label] row, scrolling the
  /// 120px-tall lazy ListView as needed (rows may sit below the fold).
  Future<void> expectListEntry(WidgetTester tester, String label) async {
    final list = find.byType(ListView);
    await tester.drag(list, const Offset(0, 500)); // back to the top
    await tester.pump();
    final entry = find.text(label);
    for (var i = 0; i < 50 && entry.evaluate().isEmpty; i++) {
      await tester.drag(list, const Offset(0, -60));
      await tester.pump(const Duration(milliseconds: 50));
    }
    expect(entry, findsWidgets, reason: 'expected a "$label" detection row');
  }

  testWidgets('boots, compiles the default engine, and renders detections', (
    tester,
  ) async {
    final status = await bootApp(tester);

    // Default engine is CompiledModel {gpu, cpu} fp16 on the Cat sample; the
    // backend in the status line must match the WebGPU capability probe.
    expect(backendOf(status), await hasWebGpu() ? 'WebGPU' : 'WASM');
    expect(find.text('CompiledModel · WebGPU+WASM · fp16'), findsOneWidget);

    // The confidence slider boots at the 55% default.
    expect(find.byType(Slider), findsOneWidget);
    expect(find.text('55%'), findsOneWidget);

    // The cats clear the default threshold; the detection list has cat rows
    // and the overlay canvas is on screen.
    expect(detectionCount(status), greaterThanOrEqualTo(1));
    expect(find.byType(ListTile), findsWidgets);
    expect(find.byType(CustomPaint), findsWidgets);
    await expectListEntry(tester, 'cat');
  }, timeout: timeout);

  testWidgets('sample picker re-runs detection on each image', (tester) async {
    var previous = await bootApp(tester);

    // Drop the threshold to its 10% minimum so the per-sample minimum counts
    // (established at the detector default of 0.25) hold with margin.
    final lowered = await setConfidence(tester, 0.0, previous);
    expect(lowered, isNotNull, reason: errorLine(tester));
    previous = lowered!;

    // Chip label, minimum detections, and an expected list entry. The file
    // names do not match their contents: Street is a kitchen with a cook,
    // Dog is a street scene with pedestrians, People is a skier.
    const cases = <(String, int, String)>[
      ('Street', 1, 'person'),
      ('Dog', 2, 'person'),
      ('People', 1, 'person'),
      ('Cat', 2, 'cat'),
    ];
    for (final (chip, minCount, entry) in cases) {
      await tester.tap(find.text(chip));
      await tester.pump();
      final status = await waitForIdle(tester, previous: previous);
      expect(
        status,
        isNotNull,
        reason: '$chip: engine failed: ${errorLine(tester)}',
      );
      expect(
        detectionCount(status!),
        greaterThanOrEqualTo(minCount),
        reason: chip,
      );
      await expectListEntry(tester, entry);
      previous = status;
    }
  }, timeout: timeout);

  testWidgets('settings dialog switches between interpreter runtimes', (
    tester,
  ) async {
    var previous = await bootApp(tester);

    // Radio tile label and the backend the status line must then report; a
    // null backend means "per the capability probe".
    final cases = <(String, String?)>[
      ('tflite-js (WASM)', 'tflite-js (WASM)'),
      ('LiteRT.js (WASM)', 'WASM'),
      ('LiteRT.js (WebGPU)', null),
    ];
    for (final (tile, backend) in cases) {
      await openSettings(tester);
      await tapInDialog(tester, 'Interpreter');
      await tapInDialog(tester, tile);
      final status = await applyAndWait(tester, previous);
      expect(
        status,
        isNotNull,
        reason: '$tile: engine failed: ${errorLine(tester)}',
      );
      final expected = backend ?? (await hasWebGpu() ? 'WebGPU' : 'WASM');
      expect(backendOf(status!), expected, reason: tile);
      expect(detectionCount(status), greaterThanOrEqualTo(1), reason: tile);
      expect(find.text('Interpreter · $tile'), findsOneWidget, reason: tile);
      previous = status;
    }
  }, timeout: timeout);

  testWidgets('settings dialog covers CompiledModel accelerators and '
      'precision', (tester) async {
    var previous = await bootApp(tester);

    // WASM (CPU): the precision tiles gray out, and the engine lands on the
    // WASM backend. The async-dispatch switch is informational and pinned on.
    await openSettings(tester);
    RadioListTile<Precision> fp16Tile() =>
        tester.widget<RadioListTile<Precision>>(
          find.widgetWithText(RadioListTile<Precision>, 'fp16 (faster)'),
        );
    expect(fp16Tile().enabled, isTrue);
    final dispatchSwitch = tester.widget<SwitchListTile>(
      find.byType(SwitchListTile),
    );
    expect(dispatchSwitch.value, isTrue);
    expect(dispatchSwitch.onChanged, isNull);

    await tapInDialog(tester, 'WASM (CPU)');
    expect(fp16Tile().enabled, isFalse);
    var status = await applyAndWait(tester, previous);
    expect(status, isNotNull, reason: errorLine(tester));
    expect(backendOf(status!), 'WASM');
    expect(find.text('CompiledModel · WASM'), findsOneWidget);
    previous = status;

    // WebGPU + WASM fallback at fp32: works everywhere, backend per probe.
    await openSettings(tester);
    await tapInDialog(tester, 'WebGPU + WASM fallback');
    await tapInDialog(tester, 'fp32 (accurate)');
    status = await applyAndWait(tester, previous);
    expect(status, isNotNull, reason: errorLine(tester));
    expect(backendOf(status!), await hasWebGpu() ? 'WebGPU' : 'WASM');
    expect(find.text('CompiledModel · WebGPU+WASM · fp32'), findsOneWidget);
    previous = status;

    // WebGPU only: real GPU where the probe says so, a surfaced engine
    // error where it does not, and the app must stay usable either way.
    await openSettings(tester);
    await tapInDialog(tester, 'WebGPU only');
    status = await applyAndWait(tester, previous);
    if (await hasWebGpu()) {
      expect(status, isNotNull, reason: errorLine(tester));
      expect(backendOf(status!), 'WebGPU');
      expect(detectionCount(status), greaterThanOrEqualTo(1));
      previous = status;
    } else {
      expect(status, isNull, reason: 'strict WebGPU must fail without GPU');
      expect(errorLine(tester), isNotNull);
    }

    // Recover back onto WASM regardless of the branch above.
    await openSettings(tester);
    await tapInDialog(tester, 'WASM (CPU)');
    status = await applyAndWait(tester, previous);
    expect(status, isNotNull, reason: errorLine(tester));
    expect(backendOf(status!), 'WASM');
    expect(detectionCount(status), greaterThanOrEqualTo(1));
  }, timeout: timeout);

  testWidgets('confidence slider re-runs detection and prunes results', (
    tester,
  ) async {
    var previous = await bootApp(tester);

    // Floor (10%): the cats plus couch clutter all clear it.
    final low = await setConfidence(tester, 0.0, previous);
    expect(low, isNotNull, reason: errorLine(tester));
    final lowCount = detectionCount(low!);
    expect(lowCount, greaterThanOrEqualTo(2));
    expect(find.text('10%'), findsOneWidget);
    previous = low;

    // Ceiling (95%): never more detections than the floor produced.
    final high = await setConfidence(tester, 1.0, previous);
    expect(high, isNotNull, reason: errorLine(tester));
    expect(detectionCount(high!), lessThanOrEqualTo(lowCount));
    expect(find.text('95%'), findsOneWidget);
    previous = high;

    // And back down: the pruning is threshold-driven, not cumulative.
    final restored = await setConfidence(tester, 0.0, previous);
    expect(restored, isNotNull, reason: errorLine(tester));
    expect(detectionCount(restored!), greaterThanOrEqualTo(2));
  }, timeout: timeout);

  testWidgets('dialog cancel leaves the engine unchanged', (tester) async {
    final status = await bootApp(tester);
    const defaultLabel = 'CompiledModel · WebGPU+WASM · fp16';
    expect(find.text(defaultLabel), findsOneWidget);

    await openSettings(tester);
    await tapInDialog(tester, 'Interpreter');
    await tapInDialog(tester, 'Cancel');
    await pumpUntil(
      tester,
      () => find.text('Apply').evaluate().isEmpty,
      'the settings dialog to close',
      limit: const Duration(seconds: 30),
    );

    // No recompile: same config label, same status line.
    expect(find.text(defaultLabel), findsOneWidget);
    expect(statusLine(tester), status);
  }, timeout: timeout);
}
