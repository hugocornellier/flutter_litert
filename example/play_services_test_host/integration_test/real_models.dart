import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';

/// Real networks and the names of their outputs, in Interpreter output order.
///
/// face_detection_short_range (BlazeFace) and mobilefacenet use convolutions,
/// depthwise convolutions, pooling, normalization and fp16 weights that are
/// dequantized at load. Each output has a reference file computed by the
/// standard LiteRT CPU runtime from the same deterministic input
/// (scripts/make_real_model_references.py), so these
/// tests check that a runtime computes what the bundled one does.
const _models = {
  'face_detection_short_range': ['regressors', 'classificators'],
  'mobilefacenet': ['embeddings'],
};

/// Largest allowed difference from the reference, as a fraction of the
/// reference's largest magnitude. Two CPU builds of LiteRT differ by about
/// 1e-5; the GPU computes in fp16.
const _cpuTolerance = 1e-3;
const _gpuTolerance = 5e-2;

/// Defines the real-model tests. [status] is read when the tests run, after
/// the runtime has been initialized.
void defineRealModelTests(LiteRtRuntimeStatus Function() status) {
  final models = <String, Uint8List>{};
  final references = <String, List<Float32List>>{};

  setUpAll(() async {
    for (final MapEntry(key: name, value: outputs) in _models.entries) {
      models[name] = (await rootBundle.load(
        'assets/$name.tflite',
      )).buffer.asUint8List();
      references[name] = [
        for (final output in outputs)
          (await rootBundle.load(
            'assets/$name.$output.f32',
          )).buffer.asFloat32List(),
      ];
    }
  });

  for (final name in _models.keys) {
    testWidgets('$name: Interpreter matches the bundled runtime', (
      tester,
    ) async {
      final interpreter = Interpreter.fromBuffer(models[name]!);
      addTearDown(interpreter.close);
      final input = interpreter.getInputTensor(0).asFloat32View();
      input.setAll(0, _patternInput(input.length));

      _report(name, 'Interpreter', () => interpreter.invoke());
      final outputs = [
        for (var i = 0; i < references[name]!.length; i++)
          interpreter.getOutputTensor(i).asFloat32View(),
      ];
      _expectClose(
        name,
        'Interpreter',
        outputs,
        references[name]!,
        _cpuTolerance,
      );
    });

    testWidgets('$name: CompiledModel on CPU matches the bundled runtime', (
      tester,
    ) async {
      final model = CompiledModel.fromBuffer(
        models[name]!,
        accelerators: {Accelerator.cpu},
      );
      addTearDown(model.close);
      _expectCompiledModel(
        name,
        'CompiledModel CPU',
        model,
        references[name]!,
        _cpuTolerance,
      );
    });

    testWidgets('$name: CompiledModel on the Play services GPU stays close '
        'to the bundled runtime', (tester) async {
      final current = status();
      if (current.runtime != LiteRtRuntime.playServices ||
          current.gpuAvailable != true) {
        markTestSkipped(
          'No Play services GPU accelerator: '
          '${current.gpuError ?? 'not built for Play services'}',
        );
        return;
      }
      // mobilefacenet has an op the GPU cannot run (L2_NORMALIZATION), which
      // the CPU picks up. The other model gets no CPU fallback, so a pass
      // proves all of it ran on the GPU.
      final model = CompiledModel.fromBuffer(
        models[name]!,
        accelerators: name == 'mobilefacenet'
            ? {Accelerator.gpu, Accelerator.cpu}
            : {Accelerator.gpu},
      );
      addTearDown(model.close);
      _expectCompiledModel(
        name,
        'CompiledModel GPU',
        model,
        references[name]!,
        _gpuTolerance,
      );
    });
  }
}

void _expectCompiledModel(
  String name,
  String api,
  CompiledModel model,
  List<Float32List> references,
  double tolerance,
) {
  final input = _patternInput(model.inputByteSizes[0] ~/ 4);
  late List<Float32List> outputs;
  _report(name, api, () => outputs = model.run([input]));
  // CompiledModel orders outputs by signature, which need not match the
  // Interpreter order; the outputs of these models all differ in length.
  final ordered = [
    for (final reference in references)
      outputs.singleWhere((output) => output.length == reference.length),
  ];
  _expectClose(name, api, ordered, references, tolerance);
}

/// The input scripts/make_real_model_references.py feeds the reference runtime: a fixed pattern in
/// [-1, 1].
Float32List _patternInput(int length) => Float32List.fromList([
  for (var i = 0; i < length; i++) ((i * 7919) % 256) / 127.5 - 1.0,
]);

/// Runs [inference] a few times and prints the median time, so a device run
/// can compare runtimes.
void _report(String name, String api, void Function() inference) {
  for (var i = 0; i < 3; i++) {
    inference();
  }
  final times = <int>[];
  for (var i = 0; i < 15; i++) {
    final stopwatch = Stopwatch()..start();
    inference();
    times.add(stopwatch.elapsedMicroseconds);
  }
  times.sort();
  debugPrint(
    'flutter_litert real model timing: $name $api '
    'median ${(times[times.length ~/ 2] / 1000).toStringAsFixed(2)} ms',
  );
}

void _expectClose(
  String name,
  String api,
  List<Float32List> outputs,
  List<Float32List> references,
  double tolerance,
) {
  for (var o = 0; o < references.length; o++) {
    final output = outputs[o];
    final reference = references[o];
    expect(output.length, reference.length);
    var maxDiff = 0.0;
    var scale = 0.0;
    for (var i = 0; i < reference.length; i++) {
      // max() propagates NaN, so a NaN output fails the check below.
      maxDiff = max(maxDiff, (output[i] - reference[i]).abs());
      scale = max(scale, reference[i].abs());
    }
    final error = maxDiff / scale;
    final label = '$name $api ${_models[name]![o]}';
    debugPrint(
      'flutter_litert real model error: $label ${error.toStringAsExponential(2)}',
    );
    expect(error, lessThan(tolerance), reason: label);
  }
}
