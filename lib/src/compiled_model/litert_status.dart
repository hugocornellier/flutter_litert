/*
 * Copyright 2026 flutter_litert authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/// Names for `LiteRtStatus` codes, so errors read as
/// `kLiteRtStatusErrorRuntimeFailure` rather than `3`.
///
/// Deliberately a Dart table rather than a binding to the runtime's own
/// `LiteRtGetStatusString`. That function is exported by the shipped macOS
/// dylib but **not** by the shipped Linux `.so`, so using it would need an
/// optional symbol lookup plus a table like this one as the fallback. Since the
/// table is required either way and produces the same strings, binding the
/// function would add platform-specific failure modes for no gain. This works
/// identically everywhere, including web, and is unit-testable with no native
/// library present.
///
/// Mirrors the enum in `litert/c/litert_common.h`. The codes are sparse by
/// design, grouped by subsystem, and an unrecognised one degrades to a plain
/// number rather than asserting: being unable to name a code is never a good
/// reason to fail while already reporting a different error.
const Map<int, String> _statusNames = {
  0: 'kLiteRtStatusOk',

  // Generic
  1: 'kLiteRtStatusErrorInvalidArgument',
  2: 'kLiteRtStatusErrorMemoryAllocationFailure',
  3: 'kLiteRtStatusErrorRuntimeFailure',
  4: 'kLiteRtStatusErrorMissingInputTensor',
  5: 'kLiteRtStatusErrorUnsupported',
  6: 'kLiteRtStatusErrorNotFound',
  7: 'kLiteRtStatusErrorTimeoutExpired',
  8: 'kLiteRtStatusErrorWrongVersion',
  9: 'kLiteRtStatusErrorUnknown',
  10: 'kLiteRtStatusErrorAlreadyExists',

  100: 'kLiteRtStatusCancelled',

  // File and serialization
  500: 'kLiteRtStatusErrorFileIO',
  501: 'kLiteRtStatusErrorInvalidFlatbuffer',
  502: 'kLiteRtStatusErrorDynamicLoading',
  503: 'kLiteRtStatusErrorSerialization',
  504: 'kLiteRtStatusErrorCompilation',

  // IR
  1000: 'kLiteRtStatusErrorIndexOOB',
  1001: 'kLiteRtStatusErrorInvalidIrType',
  1002: 'kLiteRtStatusErrorInvalidGraphInvariant',
  1003: 'kLiteRtStatusErrorGraphModification',

  // Tooling
  1500: 'kLiteRtStatusErrorInvalidToolConfig',

  // Legalization
  2000: 'kLiteRtStatusLegalizeNoMatch',
  2001: 'kLiteRtStatusErrorInvalidLegalization',

  // Transformation
  3000: 'kLiteRtStatusPatternNoMatch',
  3001: 'kLiteRtStatusInvalidTransformation',

  // Version compatibility
  4000: 'kLiteRtStatusErrorUnsupportedRuntimeVersion',
  4001: 'kLiteRtStatusErrorUnsupportedCompilerVersion',
  4002: 'kLiteRtStatusErrorIncompatibleByteCodeVersion',

  // Shape inference
  5000: 'kLiteRtStatusErrorUnsupportedOpShapeInferer',
  5001: 'kLiteRtStatusErrorShapeInferenceFailed',
};

/// The name for [status], or null when it is not a code this version knows.
String? liteRtStatusName(int status) => _statusNames[status];

/// Formats [status] for an error message, as `3 (kLiteRtStatusErrorRuntimeFailure)`.
///
/// Keeps the number as well as the name, because the number is what appears in
/// LiteRT's own logging and in upstream bug reports, so dropping it would make
/// our errors harder to correlate rather than easier.
///
/// An unknown code renders as `42 (unrecognised LiteRtStatus)`.
String describeLiteRtStatus(int status) {
  final name = liteRtStatusName(status);
  return name == null
      ? '$status (unrecognised LiteRtStatus)'
      : '$status ($name)';
}
