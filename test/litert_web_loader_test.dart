@TestOn('browser')
library;

import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';

/// Contract test for [configureLiteRtWebLoader]: a misconfigured module URL
/// must surface as a clear [StateError] rather than a hang or a silent CDN
/// fallback.
///
/// Lives in its own file on purpose: the LiteRT.js loader is once-per-page
/// (a failed load marks window.LiteRtReady = false permanently), so this
/// suite deliberately poisons its page and cannot share one with tests that
/// need a working runtime. Suites in other files get their own page.
///
/// The same trivial float32 add model as compiled_model_web_test.dart,
/// embedded because browser tests cannot read the file system.
const _addModelBase64 =
    'JAAAAFRGTDMAAAAAAAAAABQAGAAEAAgADAAAABAAAAAAABQAFAAAAAMAAADkAQAAmAAAAIAAAAAE'
    'AAAAAQAAABAAAAAAAAoAEAAEAAgADAAKAAAAPAAAABwAAAAEAAAADwAAAHNlcnZpbmdfZGVmYXVs'
    'dAABAAAABAAAAOT///8IAAAAAgAAAAEAAAB4AAAAAQAAAAwAAAAIAAwABAAIAAgAAAAIAAAAAQAA'
    'AAEAAABhAAAAAQAAAAQAAACk/v//AAAAAAAAAAABAAAAEAAAAAwAFAAEAAgADAAQAAwAAACUAAAA'
    'iAAAAHwAAAAEAAAAAgAAAEQAAAAEAAAA0v///wAAAAsYAAAADAAAAAQAAAD4/v//AQAAAAIAAAAC'
    'AAAAAAAAAAEAAAAAAA4AFAAAAAgADAAHABAADgAAAAAAAAsYAAAADAAAAAQAAAA0////AQAAAAAA'
    'AAACAAAAAQAAAAEAAAABAAAAAgAAAAEAAAABAAAAAwAAAHAAAAA0AAAABAAAAKj///8UAAAABAAA'
    'AAYAAABvdXRwdXQAAAQAAAABAAAACAAAAAgAAAADAAAA1P///xQAAAAEAAAABQAAAGlucHV0AAAA'
    'BAAAAAEAAAAIAAAACAAAAAMAAAAMAAwABAAAAAAACAAMAAAAEAAAAAQAAAADAAAAYWRkAAQAAAAB'
    'AAAACAAAAAgAAAADAAAAAQAAAAgAAAAEAAQABAAAAA==';

void main() {
  test('a bogus self-hosted module URL fails fast with a StateError', () async {
    final Uint8List modelBytes = base64Decode(_addModelBase64);

    // Port 1 is never listening; the dynamic import() rejects immediately,
    // the injected loader records the error, and every runtime entry point
    // must then throw instead of hanging until the 30s load timeout.
    configureLiteRtWebLoader(
      moduleUrl: 'http://127.0.0.1:1/litert-core-missing.mjs',
      wasmUrl: 'http://127.0.0.1:1/wasm/',
    );

    final sw = Stopwatch()..start();
    await expectLater(
      CompiledModel.fromBufferAsync(modelBytes),
      throwsA(
        isA<StateError>().having(
          (e) => e.message,
          'message',
          contains('LiteRT.js'),
        ),
      ),
    );
    sw.stop();
    expect(
      sw.elapsed,
      lessThan(const Duration(seconds: 25)),
      reason:
          'a failed load must reject via the loader error, not the '
          'load timeout',
    );

    // The failure is sticky for the page: later calls fail the same way.
    await expectLater(
      CompiledModel.fromBufferAsync(modelBytes),
      throwsA(isA<StateError>()),
    );
  });
}
