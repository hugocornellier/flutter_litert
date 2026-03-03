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

import '../web/delegate.dart';

/// FlexDelegate (no-op on web)
class FlexDelegate extends Delegate {
  FlexDelegate();

  /// Always returns false on web.
  static bool get isAvailable => false;

  /// Async constructor (returns immediately on web).
  static Future<FlexDelegate> create() async => FlexDelegate();

  @override
  void delete() {}
}
