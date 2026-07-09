import 'package:flutter/material.dart';
import 'package:flutter_litert/flutter_litert.dart';

import 'detector.dart';

void main() {
  runApp(const WebExampleApp());
}

/// Web mirror of the native example's object-detection demo.
///
/// Same shape as the native app: a sample-image picker, an inference-engine
/// settings dialog, and a detection overlay with per-box colors and a
/// microsecond timing readout. The engines are the web stacks: CompiledModel
/// (LiteRT.js), LiteRtInterpreter (LiteRT.js), and Interpreter (tflite-js).
class WebExampleApp extends StatelessWidget {
  const WebExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'flutter_litert web example',
      theme: ThemeData(colorSchemeSeed: Colors.teal),
      home: const _DetectionDemo(),
    );
  }
}

const _kSamples = <(String, String)>[
  ('Street', 'assets/street.jpg'),
  ('Cat', 'assets/cat.jpg'),
  ('Dog', 'assets/dog.jpg'),
  ('People', 'assets/people.jpg'),
];

class _DetectionDemo extends StatefulWidget {
  const _DetectionDemo();

  @override
  State<_DetectionDemo> createState() => _DetectionDemoState();
}

class _DetectionDemoState extends State<_DetectionDemo> {
  WebEngineConfig _cfg = const WebEngineConfig();
  Detector? _detector;
  DetectionResult? _result;
  String _sample = _kSamples[1].$2; // Cat
  double _confThres = 0.55;
  String? _status;
  Object? _error;
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _rebuildDetector();
  }

  @override
  void dispose() {
    _detector?.close();
    super.dispose();
  }

  Future<void> _rebuildDetector() async {
    setState(() {
      _busy = true;
      _error = null;
      _result = null;
      _status = 'Compiling ${_cfg.label}...';
    });
    _detector?.close();
    _detector = null;
    try {
      final detector = await Detector.create(config: _cfg);
      if (!mounted) {
        detector.close();
        return;
      }
      _detector = detector;
      await _detect();
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _busy = false;
        _error = e;
        _status = null;
      });
    }
  }

  Future<void> _detect() async {
    final detector = _detector;
    if (detector == null) return;
    setState(() {
      _busy = true;
      _error = null;
      _status = 'Running...';
    });
    try {
      final result = await detector.detectAsset(_sample, confThres: _confThres);
      if (!mounted) return;
      setState(() {
        _busy = false;
        _result = result;
        _status =
            '${detector.backend} · ${result.inferenceMicros}µs · '
            '${result.detections.length} detections';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _busy = false;
        _error = e;
        _status = null;
      });
    }
  }

  Future<void> _openSettings() async {
    final next = await showDialog<WebEngineConfig>(
      context: context,
      builder: (_) => _EngineSettingsDialog(initial: _cfg),
    );
    if (next == null) return;
    setState(() => _cfg = next);
    await _rebuildDetector();
  }

  @override
  Widget build(BuildContext context) {
    final result = _result;
    return Scaffold(
      appBar: AppBar(
        title: const Text('flutter_litert web example'),
        actions: [
          TextButton.icon(
            onPressed: _busy ? null : _openSettings,
            icon: const Icon(Icons.tune),
            label: Text(_cfg.label),
          ),
        ],
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
            child: Wrap(
              spacing: 8,
              children: [
                for (final (label, path) in _kSamples)
                  ChoiceChip(
                    label: Text(label),
                    selected: _sample == path,
                    onSelected: _busy
                        ? null
                        : (sel) {
                            if (!sel) return;
                            setState(() => _sample = path);
                            _detect();
                          },
                  ),
              ],
            ),
          ),
          // Confidence threshold slider, mirroring the native example's.
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Row(
              children: [
                const Text('Confidence', style: TextStyle(fontSize: 12)),
                Expanded(
                  child: Slider(
                    value: _confThres,
                    min: 0.1,
                    max: 0.95,
                    divisions: 17,
                    label: '${(_confThres * 100).round()}%',
                    onChanged: _busy
                        ? null
                        : (v) => setState(() => _confThres = v),
                    onChangeEnd: _busy ? null : (_) => _detect(),
                  ),
                ),
                Text(
                  '${(_confThres * 100).round()}%',
                  style: const TextStyle(fontSize: 12),
                ),
              ],
            ),
          ),
          if (_status != null)
            Padding(
              padding: const EdgeInsets.only(bottom: 6),
              child: Text(_status!),
            ),
          if (_error != null)
            Padding(
              padding: const EdgeInsets.all(8),
              child: Text(
                'Engine failed: $_error',
                style: TextStyle(color: Theme.of(context).colorScheme.error),
              ),
            ),
          Expanded(
            child: result == null
                ? const Center(child: CircularProgressIndicator())
                : FittedBox(
                    child: SizedBox(
                      width: result.image.width.toDouble(),
                      height: result.image.height.toDouble(),
                      child: CustomPaint(painter: _DetectionPainter(result)),
                    ),
                  ),
          ),
          if (result != null)
            SizedBox(
              height: 120,
              child: ListView(
                children: [
                  for (final (i, d) in result.detections.indexed)
                    ListTile(
                      dense: true,
                      leading: Icon(Icons.square, color: _detectionColor(i)),
                      title: Text(cocoLabels[d.cls]),
                      trailing: Text(d.score.toStringAsFixed(2)),
                    ),
                ],
              ),
            ),
        ],
      ),
    );
  }
}

/// Engine settings dialog mirroring the native example's, with web knobs:
/// the Interpreter engine picks a runtime, the CompiledModel engine picks
/// WASM/WebGPU accelerators and (on WebGPU) precision.
class _EngineSettingsDialog extends StatefulWidget {
  const _EngineSettingsDialog({required this.initial});

  final WebEngineConfig initial;

  @override
  State<_EngineSettingsDialog> createState() => _EngineSettingsDialogState();
}

class _EngineSettingsDialogState extends State<_EngineSettingsDialog> {
  late WebEngineConfig _cfg = widget.initial;

  static String _accelKey(Set<Accelerator> accels) {
    final sorted = accels.toList()..sort((a, b) => a.name.compareTo(b.name));
    return sorted.map((e) => e.name).join('+');
  }

  Widget _heading(String text) => Padding(
    padding: const EdgeInsets.only(top: 4, bottom: 4, left: 8),
    child: Text(
      text,
      style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 12),
    ),
  );

  @override
  Widget build(BuildContext context) {
    final bool gpuOn = _cfg.accelerators.contains(Accelerator.gpu);
    return AlertDialog(
      title: const Text('Inference engine'),
      content: SizedBox(
        width: 380,
        child: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              SegmentedButton<WebEngineKind>(
                segments: const [
                  ButtonSegment(
                    value: WebEngineKind.interpreter,
                    label: Text('Interpreter'),
                  ),
                  ButtonSegment(
                    value: WebEngineKind.compiledModel,
                    label: Text('CompiledModel'),
                  ),
                ],
                selected: {_cfg.kind},
                onSelectionChanged: (sel) =>
                    setState(() => _cfg = _cfg.copyWith(kind: sel.first)),
              ),
              const Divider(height: 24),
              if (_cfg.kind == WebEngineKind.interpreter)
                ..._interpreterOptions()
              else
                ..._compiledModelOptions(gpuOn),
            ],
          ),
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Cancel'),
        ),
        FilledButton(
          onPressed: () => Navigator.pop(context, _cfg),
          child: const Text('Apply'),
        ),
      ],
    );
  }

  List<Widget> _interpreterOptions() => [
    _heading('Runtime'),
    RadioGroup<WebInterpreterRuntime>(
      groupValue: _cfg.runtime,
      onChanged: (v) {
        if (v != null) setState(() => _cfg = _cfg.copyWith(runtime: v));
      },
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          for (final r in WebInterpreterRuntime.values)
            RadioListTile<WebInterpreterRuntime>(
              dense: true,
              value: r,
              title: Text(interpreterRuntimeLabel(r)),
              subtitle: r == WebInterpreterRuntime.liteRtWebGpu
                  ? const Text(
                      'Falls back to WASM without WebGPU',
                      style: TextStyle(fontSize: 11),
                    )
                  : null,
            ),
        ],
      ),
    ),
  ];

  List<Widget> _compiledModelOptions(bool gpuOn) {
    final String curKey = _accelKey(_cfg.accelerators);

    Widget accel(String label, Set<Accelerator> value, {String? subtitle}) =>
        RadioListTile<String>(
          dense: true,
          value: _accelKey(value),
          title: Text(label),
          subtitle: subtitle == null
              ? null
              : Text(subtitle, style: const TextStyle(fontSize: 11)),
        );

    return [
      _heading('Accelerator'),
      RadioGroup<String>(
        groupValue: curKey,
        onChanged: (key) {
          const map = {
            'cpu': {Accelerator.cpu},
            'gpu': {Accelerator.gpu},
            'cpu+gpu': {Accelerator.cpu, Accelerator.gpu},
          };
          final accels = map[key];
          if (accels != null) {
            setState(() => _cfg = _cfg.copyWith(accelerators: accels));
          }
        },
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            accel('WASM (CPU)', const {Accelerator.cpu}),
            accel('WebGPU + WASM fallback', const {
              Accelerator.cpu,
              Accelerator.gpu,
            }),
            accel('WebGPU only', const {
              Accelerator.gpu,
            }, subtitle: 'Fails on browsers without WebGPU'),
          ],
        ),
      ),
      const Divider(height: 16),
      _heading('Precision (WebGPU only)'),
      RadioGroup<Precision>(
        groupValue: _cfg.precision,
        onChanged: (v) {
          if (v != null) setState(() => _cfg = _cfg.copyWith(precision: v));
        },
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            RadioListTile<Precision>(
              dense: true,
              value: Precision.fp16,
              enabled: gpuOn,
              title: const Text('fp16 (faster)'),
            ),
            RadioListTile<Precision>(
              dense: true,
              value: Precision.fp32,
              enabled: gpuOn,
              title: const Text('fp32 (accurate)'),
            ),
          ],
        ),
      ),
      const SwitchListTile(
        dense: true,
        value: true,
        title: Text('Async dispatch (runAsync)'),
        subtitle: Text(
          'Web inference is always async',
          style: TextStyle(fontSize: 11),
        ),
        onChanged: null,
      ),
    ];
  }
}

// Per-detection overlay colors, cycled by index so adjacent boxes (and their
// list icons) stay visually distinct, matching the native example's palette.
const List<Color> _detectionPalette = [
  Color(0xFF00E5FF), // cyan
  Color(0xFFFF5252), // red
  Color(0xFF69F0AE), // green
  Color(0xFFFFD740), // amber
  Color(0xFFFF4081), // pink
  Color(0xFF7C4DFF), // deep purple
  Color(0xFF40C4FF), // light blue
  Color(0xFFB2FF59), // lime
  Color(0xFFFF6E40), // deep orange
  Color(0xFFEEFF41), // yellow
  Color(0xFF1DE9B6), // teal
  Color(0xFFE040FB), // purple
];

Color _detectionColor(int i) => _detectionPalette[i % _detectionPalette.length];

Color _onColor(Color bg) =>
    bg.computeLuminance() > 0.5 ? Colors.black : Colors.white;

class _DetectionPainter extends CustomPainter {
  _DetectionPainter(this.result);

  final DetectionResult result;

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawImage(result.image, Offset.zero, Paint());
    for (final (i, d) in result.detections.indexed) {
      final color = _detectionColor(i);
      final b = d.bboxXYXY;
      canvas.drawRect(
        Rect.fromLTRB(b[0], b[1], b[2], b[3]),
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3
          ..color = color,
      );
      final label = TextPainter(
        text: TextSpan(
          text: '${cocoLabels[d.cls]} ${d.score.toStringAsFixed(2)}',
          style: TextStyle(
            color: _onColor(color),
            fontSize: 16,
            backgroundColor: color,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      label.paint(canvas, Offset(b[0], (b[1] - 20).clamp(0, size.height)));
    }
  }

  @override
  bool shouldRepaint(_DetectionPainter oldDelegate) =>
      oldDelegate.result != result;
}
