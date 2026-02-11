// Copyright (c) 2019, the Dart project authors. Please see the AUTHORS file
// for details. All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

/// LiteRT (formerly TensorFlow Lite) for Flutter
library flutter_litert;

import 'package:ffi/ffi.dart';
import 'package:flutter_litert/src/bindings/bindings.dart';

export 'src/delegate.dart';
export 'src/delegates/gpu_delegate.dart';
export 'src/delegates/metal_delegate.dart';
export 'src/delegates/xnnpack_delegate.dart';
export 'src/delegates/coreml_delegate.dart';
export 'src/interpreter.dart';
export 'src/interpreter_options.dart';
export 'src/isolate_interpreter.dart';
export 'src/quanitzation_params.dart';
export 'src/tensor.dart';
export 'src/util/byte_conversion_utils.dart';
export 'src/util/list_shape_extension.dart';
export 'src/custom_ops/transpose_conv_bias.dart';

/// LiteRT version information.
String get version => tfliteBinding.TfLiteVersion().cast<Utf8>().toDartString();
