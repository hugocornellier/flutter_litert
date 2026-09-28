import 'dart:typed_data';

import 'compiled_model/compiled_model.dart';
import 'util/async_lock.dart';

/// One pool slot: a [CompiledModel], a reusable input buffer, and a lock that
/// serializes that slot's calls.
class _CompiledSlot {
  final CompiledModel model;
  final Float32List input;
  final AsyncLock lock = AsyncLock();

  _CompiledSlot(this.model, this.input);
}

/// A round-robin pool of [CompiledModel] instances for the LiteRT Next engine.
///
/// A single [CompiledModel] is not safe to drive from overlapping calls, and its
/// scratch input buffer must not be overwritten while an inference is still
/// reading it. Each slot therefore bundles its own model, its own reusable input
/// [Float32List], and an [AsyncLock] that serializes that slot's calls.
///
/// [withModel] hands work to the next slot round-robin, so concurrent calls,
/// such as one per detected hand/person dispatched via `Future.wait`, land on
/// distinct slots and overlap, while two calls colliding on the same slot run
/// back-to-back.
///
/// A pool of size 1 is the safe degenerate case: every call serializes through
/// the single slot's lock and shares its one buffer, exactly equivalent to a
/// single [CompiledModel] guarded by a lock.
class CompiledModelPool {
  final List<_CompiledSlot> _slots = [];
  int _next = 0;

  /// True once [initialize] has built at least one slot.
  bool get isInitialized => _slots.isNotEmpty;

  /// Number of slots (models) in the pool.
  int get size => _slots.length;

  /// Builds [poolSize] slots (clamped to at least 1), disposing any existing
  /// ones first.
  ///
  /// [create] is called once per slot to produce a fresh [CompiledModel]; each
  /// slot also allocates a reusable input buffer of [inputFloats] float32
  /// values. [onFirstModel], if given, is called with the first model built;
  /// use it to resolve I/O tensor indices once.
  void initialize({
    required int poolSize,
    required int inputFloats,
    required CompiledModel Function() create,
    void Function(CompiledModel firstModel)? onFirstModel,
  }) {
    dispose();
    final int n = poolSize < 1 ? 1 : poolSize;
    try {
      for (int i = 0; i < n; i++) {
        final CompiledModel model = create();
        try {
          if (i == 0) onFirstModel?.call(model);
        } catch (_) {
          model.close();
          rethrow;
        }
        _slots.add(_CompiledSlot(model, Float32List(inputFloats)));
      }
    } catch (_) {
      // Close any slots already built so a failed setup (e.g. an unsupported
      // model rejected by onFirstModel) leaves no leaked native models, and the
      // caller can fall back to another engine.
      dispose();
      rethrow;
    }
  }

  /// Runs [task] on the next slot (round-robin), serialized against other calls
  /// on that same slot.
  ///
  /// [task] receives the slot's [CompiledModel] and its reusable input
  /// [Float32List]: fill the buffer, then call `model.runAsync([input])`.
  ///
  /// Throws [StateError] if the pool is not initialized.
  Future<T> withModel<T>(
    Future<T> Function(CompiledModel model, Float32List input) task,
  ) {
    if (_slots.isEmpty) {
      throw StateError('CompiledModelPool not initialized.');
    }
    final _CompiledSlot slot = _slots[_next];
    _next = (_next + 1) % _slots.length;
    return slot.lock.run(() => task(slot.model, slot.input));
  }

  /// Closes every [CompiledModel] and clears all slots.
  void dispose() {
    for (final _CompiledSlot slot in _slots) {
      slot.model.close();
    }
    _slots.clear();
    _next = 0;
  }
}
