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
import '../web/delegate.dart';

/// Metal Delegate for iOS/macOS (no-op on web)
class GpuDelegate extends Delegate {
  GpuDelegate({GpuDelegateOptions? options});

  @override
  void delete() {}

  /// Binds a Metal buffer to an input or output tensor (no-op on web).
  bool bindMetalBufferToTensor(int tensorIndex, int metalBuffer) => false;
}

/// Metal Delegate options (no-op on web)
class GpuDelegateOptions {
  GpuDelegateOptions({
    bool allowPrecisionLoss = false,
    int waitType = 0,
    bool enableQuantization = true,
  });

  void delete() {}
}
