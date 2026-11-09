import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:flutter_litert/native.dart';
import 'package:flutter_litert/src/bindings/tensorflow_lite_bindings_generated.dart';
import 'package:flutter_test/flutter_test.dart';

/// Guards the Dart layout of `TfLiteXNNPackDelegateOptions`.
///
/// LiteRT 2.3.2, which runs the Interpreter on Android, appended
/// `bool weight_cache_lock_memory` after `weight_cache_provider`, and its
/// delegate constructor copies the whole struct. When the Dart struct stopped
/// at `weight_cache_provider` it was 56 bytes and native read 64, so the copy
/// ran 8 bytes past the calloc'd allocation. These offsets are the C layout of
/// `tflite/delegates/xnnpack/xnnpack_delegate.h` at v2.3.2 on LP64 and LLP64.
void main() {
  final is64Bit = sizeOf<IntPtr>() == 8;
  final skip64 = is64Bit ? false : 'offsets below are for 64-bit hosts';

  test('struct is 64 bytes, matching LiteRT 2.3.2', () {
    expect(sizeOf<TfLiteXNNPackDelegateOptions>(), 64);
  }, skip: skip64);

  test('each field lands at its C offset', () {
    final options = calloc<TfLiteXNNPackDelegateOptions>();
    addTearDown(() => calloc.free(options));
    final bytes = options.cast<Uint8>().asTypedList(
      sizeOf<TfLiteXNNPackDelegateOptions>(),
    );

    options.ref
      ..num_threads = 0x11111111
      ..runtime_flags = 0x22222222
      ..flags = 0x33333333
      ..weights_cache = Pointer.fromAddress(0x4444444444444444)
      ..handle_variable_ops = true
      ..weight_cache_file_path = Pointer.fromAddress(0x5555555555555555)
      ..weight_cache_file_descriptor = 0x66666666
      ..weight_cache_provider = Pointer.fromAddress(0x7777777777777777)
      ..weight_cache_lock_memory = true;

    int u32(int offset) =>
        (options.cast<Uint8>() + offset).cast<Uint32>().value;
    int u64(int offset) =>
        (options.cast<Uint8>() + offset).cast<Uint64>().value;

    expect(u32(0), 0x11111111);
    expect(u32(4), 0x22222222);
    expect(u32(8), 0x33333333);
    expect(u64(16), 0x4444444444444444);
    expect(bytes[24], 1);
    expect(u64(32), 0x5555555555555555);
    expect(u32(40), 0x66666666);
    expect(u64(48), 0x7777777777777777);
    expect(bytes[56], 1);
    // Padding after the trailing bool stays untouched.
    expect(bytes.sublist(57), everyElement(0));
  }, skip: skip64);

  test('XNNPackDelegateOptions leaves weight_cache_lock_memory false', () {
    final options = XNNPackDelegateOptions(numThreads: 2);
    addTearDown(options.delete);

    expect(options.base.ref.weight_cache_lock_memory, isFalse);
    expect(options.base.ref.num_threads, 2);
    expect(
      options.base.ref.flags,
      TFLITE_XNNPACK_DELEGATE_FLAG_QS8 | TFLITE_XNNPACK_DELEGATE_FLAG_QU8,
    );
  });

  test('host runtime accepts the larger options struct', () {
    final options = XNNPackDelegateOptions(numThreads: 2);
    final delegate = XNNPackDelegate(options: options);
    addTearDown(() {
      delegate.delete();
      options.delete();
    });

    expect(delegate.base, isNot(nullptr));
  });
}
