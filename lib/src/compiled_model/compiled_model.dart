export 'compiled_model_unsupported.dart'
    if (dart.library.io) 'compiled_model_native.dart'
    if (dart.library.js_interop) 'compiled_model_web.dart';
