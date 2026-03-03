/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

/// Metal Delegate for iOS/macOS (unsupported platform)
class GpuDelegate {
  GpuDelegate({GpuDelegateOptions? options}) {
    throw UnsupportedError('GpuDelegate is not supported on this platform');
  }

  void delete() {
    throw UnsupportedError('GpuDelegate is not supported on this platform');
  }
}

/// Metal Delegate options (unsupported platform)
class GpuDelegateOptions {
  GpuDelegateOptions({
    bool allowPrecisionLoss = false,
    int waitType = 0,
    bool enableQuantization = true,
  }) {
    throw UnsupportedError(
      'GpuDelegateOptions is not supported on this platform',
    );
  }

  void delete() {
    throw UnsupportedError(
      'GpuDelegateOptions is not supported on this platform',
    );
  }
}
