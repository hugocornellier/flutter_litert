/*
 * Copyright 2025 flutter_litert authors. All Rights Reserved.
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

/// FlexDelegate (unsupported platform)
class FlexDelegate {
  FlexDelegate() {
    throw UnsupportedError('FlexDelegate is not supported on this platform');
  }

  /// Always returns false on unsupported platforms.
  static bool get isAvailable => false;

  /// Throws on unsupported platforms.
  static Future<FlexDelegate> create() async {
    throw UnsupportedError('FlexDelegate is not supported on this platform');
  }

  void delete() {
    throw UnsupportedError('FlexDelegate is not supported on this platform');
  }
}
