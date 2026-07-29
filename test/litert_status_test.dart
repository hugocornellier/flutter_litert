import 'package:flutter_litert/src/compiled_model/litert_status.dart';
import 'package:flutter_test/flutter_test.dart';

/// Pure Dart, so this runs on every platform with no LiteRT library present.
/// That portability is the reason the table exists instead of a binding to
/// `LiteRtGetStatusString`, which the shipped Linux `.so` does not export.
void main() {
  test('names the codes this project actually hits', () {
    // 3 is the one that mattered: it is what CompiledModel returns for the
    // upstream dynamic-output defect, and "LiteRtStatus=3" was unreadable.
    expect(liteRtStatusName(3), 'kLiteRtStatusErrorRuntimeFailure');
    expect(liteRtStatusName(0), 'kLiteRtStatusOk');
    expect(liteRtStatusName(1), 'kLiteRtStatusErrorInvalidArgument');
    expect(liteRtStatusName(2), 'kLiteRtStatusErrorMemoryAllocationFailure');
    expect(liteRtStatusName(5), 'kLiteRtStatusErrorUnsupported');
  });

  test('covers the sparse high ranges, not just 0-10', () {
    expect(liteRtStatusName(100), 'kLiteRtStatusCancelled');
    expect(liteRtStatusName(500), 'kLiteRtStatusErrorFileIO');
    expect(liteRtStatusName(504), 'kLiteRtStatusErrorCompilation');
    expect(liteRtStatusName(1000), 'kLiteRtStatusErrorIndexOOB');
    expect(liteRtStatusName(1500), 'kLiteRtStatusErrorInvalidToolConfig');
    expect(liteRtStatusName(2000), 'kLiteRtStatusLegalizeNoMatch');
    expect(liteRtStatusName(3001), 'kLiteRtStatusInvalidTransformation');
    expect(
      liteRtStatusName(4002),
      'kLiteRtStatusErrorIncompatibleByteCodeVersion',
    );
    expect(liteRtStatusName(5001), 'kLiteRtStatusErrorShapeInferenceFailed');
  });

  test('returns null for codes it does not know', () {
    // Gaps between the subsystem ranges are genuinely unassigned.
    expect(liteRtStatusName(11), isNull);
    expect(liteRtStatusName(42), isNull);
    expect(liteRtStatusName(999), isNull);
    expect(liteRtStatusName(-1), isNull);
  });

  test('keeps the number alongside the name', () {
    // The number is what appears in LiteRT's own logging and in upstream bug
    // reports, so dropping it would make errors harder to correlate.
    expect(describeLiteRtStatus(3), '3 (kLiteRtStatusErrorRuntimeFailure)');
    expect(describeLiteRtStatus(0), '0 (kLiteRtStatusOk)');
  });

  test('degrades gracefully on an unknown code', () {
    // Failing to name a code must never break the error being reported.
    expect(describeLiteRtStatus(42), '42 (unrecognised LiteRtStatus)');
    expect(() => describeLiteRtStatus(-7), returnsNormally);
  });
}
