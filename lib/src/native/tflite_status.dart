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

/// Names for `TfLiteStatus` codes returned by the Interpreter C API.
///
/// Keeping the numeric value makes the error directly searchable in upstream
/// logs while the symbolic name makes it understandable without consulting a
/// header. Unknown future values remain reportable.
const Map<int, String> _statusNames = {
  0: 'kTfLiteOk',
  1: 'kTfLiteError',
  2: 'kTfLiteDelegateError',
  3: 'kTfLiteApplicationError',
  4: 'kTfLiteDelegateDataNotFound',
  5: 'kTfLiteDelegateDataWriteError',
  6: 'kTfLiteDelegateDataReadError',
  7: 'kTfLiteUnresolvedOps',
  8: 'kTfLiteCancelled',
  9: 'kTfLiteOutputShapeNotKnown',
};

/// The symbolic name for [status], or null when this package does not know it.
String? tfLiteStatusName(int status) => _statusNames[status];

/// Formats [status] as `1 (kTfLiteError)`.
String describeTfLiteStatus(int status) {
  final name = tfLiteStatusName(status);
  return name == null
      ? '$status (unrecognised TfLiteStatus)'
      : '$status ($name)';
}

/// Throws a contextual [StateError] when [status] is not `kTfLiteOk`.
void checkTfLiteStatus(String operation, int status) {
  if (status != 0) {
    throw StateError(
      '$operation failed with TfLiteStatus=${describeTfLiteStatus(status)}.',
    );
  }
}
