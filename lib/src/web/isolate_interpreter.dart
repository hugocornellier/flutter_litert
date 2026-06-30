import 'dart:async';

import '../isolate_interpreter_state.dart';
import 'interpreter.dart';

/// Web implementation of IsolateInterpreter.
///
/// On web, isolates are not available. This wraps the regular Interpreter
/// and runs inference on the main thread.
class IsolateInterpreter {
  final Interpreter _interpreter;
  IsolateInterpreterState _state = IsolateInterpreterState.idle;
  final StreamController<IsolateInterpreterState> _stateController =
      StreamController<IsolateInterpreterState>.broadcast();

  IsolateInterpreter._({
    required this.address,
    required this.debugName,
    required Interpreter interpreter,
  }) : _interpreter = interpreter;

  /// Address-based factory, not supported on web.
  ///
  /// Web tensors are Dart-side buffers with no native address, so there is
  /// nothing to construct from an [address]. Throws [UnsupportedError]; use
  /// [createFromInterpreter] to wrap an existing [Interpreter] instead.
  static Future<IsolateInterpreter> create({
    required int address,
    String debugName = 'TfLiteInterpreterIsolate',
  }) async {
    // On web, address-based construction is not supported.
    // Callers should use createFromInterpreter instead.
    throw UnsupportedError(
      'IsolateInterpreter.create with address is not supported on web. '
      'Use the interpreter directly.',
    );
  }

  /// Creates a web IsolateInterpreter from an existing Interpreter.
  static Future<IsolateInterpreter> createFromInterpreter(
    Interpreter interpreter, {
    String debugName = 'TfLiteInterpreterWeb',
  }) async {
    return IsolateInterpreter._(
      address: 0,
      debugName: debugName,
      interpreter: interpreter,
    );
  }

  /// The interpreter address (always 0 on web).
  final int address;

  /// Debug label for this interpreter.
  final String debugName;

  /// Stream of interpreter state changes.
  Stream<IsolateInterpreterState> get stateChanges => _stateController.stream;

  /// The current interpreter state.
  IsolateInterpreterState get state => _state;

  /// Run LiteRT model for single input and output.
  Future<void> run(Object input, Object output) async {
    _state = IsolateInterpreterState.loading;
    _stateController.add(_state);
    try {
      _interpreter.run(input, output);
    } finally {
      _state = IsolateInterpreterState.idle;
      _stateController.add(_state);
    }
  }

  /// Run LiteRT model for multiple inputs and outputs.
  Future<void> runForMultipleInputs(
    List<Object> inputs,
    Map<int, Object> outputs,
  ) async {
    _state = IsolateInterpreterState.loading;
    _stateController.add(_state);
    try {
      _interpreter.runForMultipleInputs(inputs, outputs);
    } finally {
      _state = IsolateInterpreterState.idle;
      _stateController.add(_state);
    }
  }

  /// Close resources.
  Future<void> close() async {
    _interpreter.close();
    await _stateController.close();
  }
}
