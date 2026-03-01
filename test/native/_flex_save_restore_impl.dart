// Implementation for save/restore checkpoint integration tests.
//
// This file is run as a subprocess by flex_save_restore_test.dart.
// It may crash on exit (SIGBUS from TF atexit handlers) — that is expected
// and handled by the parent wrapper.
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_litert/flutter_litert.dart';

File get _flexModelFile =>
    File('${Directory.current.path}/test/assets/training_model_flex.tflite');

void main() {
  test('save/restore signatures and checkpoint round-trip', () async {
    var flex = FlexDelegate();
    var opts = InterpreterOptions();
    opts.addDelegate(flex);
    var interp = Interpreter.fromFile(_flexModelFile, options: opts);

    // --- Verify save/restore signature tensor types ---
    final save = interp.getSignatureRunner('save');
    expect(save.inputNames, contains('checkpoint_path'));
    expect(save.getInputTensor('checkpoint_path').type, TensorType.string);
    save.close();

    final restore = interp.getSignatureRunner('restore');
    expect(restore.inputNames, contains('checkpoint_path'));
    expect(restore.getInputTensor('checkpoint_path').type, TensorType.string);
    restore.close();

    // --- Train 10 steps, save checkpoint A ---
    final tmpDir = Directory.systemTemp.createTempSync('litert_ckpt_');
    final train = interp.getSignatureRunner('train');
    final loss = Float32List(1);
    for (var i = 0; i < 10; i++) {
      train.run(
        {
          'x': [
            [1.0],
          ],
          'y': [
            [2.0],
          ],
        },
        {'loss': loss},
      );
    }

    final inferA = interp.getSignatureRunner('infer');
    final predA = [
      [0.0],
    ];
    inferA.run(
      {
        'x': [
          [1.0],
        ],
      },
      {'output': predA},
    );
    inferA.close();

    final ckptA = '${tmpDir.path}/a.ckpt';
    final saveA = interp.getSignatureRunner('save');
    final saveStatusA = [0];
    saveA.run(
      {
        'checkpoint_path': [ckptA],
      },
      {'status': saveStatusA},
    );
    saveA.close();
    expect(saveStatusA[0], equals(0));

    // Verify checkpoint files were created.
    expect(
      File('$ckptA.index').existsSync(),
      isTrue,
      reason: 'Checkpoint .index file should exist',
    );
    expect(
      File('$ckptA.data-00000-of-00001').existsSync(),
      isTrue,
      reason: 'Checkpoint .data file should exist',
    );
    expect(File('$ckptA.index').lengthSync(), greaterThan(0));
    expect(File('$ckptA.data-00000-of-00001').lengthSync(), greaterThan(0));

    // --- Train 40 more steps, save checkpoint B ---
    for (var i = 0; i < 40; i++) {
      train.run(
        {
          'x': [
            [1.0],
          ],
          'y': [
            [2.0],
          ],
        },
        {'loss': loss},
      );
    }
    train.close();

    final inferB = interp.getSignatureRunner('infer');
    final predB = [
      [0.0],
    ];
    inferB.run(
      {
        'x': [
          [1.0],
        ],
      },
      {'output': predB},
    );
    inferB.close();

    // More training → prediction closer to target.
    expect(predB[0][0], greaterThan(predA[0][0]));

    final ckptB = '${tmpDir.path}/b.ckpt';
    final saveB = interp.getSignatureRunner('save');
    saveB.run(
      {
        'checkpoint_path': [ckptB],
      },
      {
        'status': [0],
      },
    );
    saveB.close();

    // --- Fresh interpreter → restore checkpoint A ---
    interp.close();
    flex.delete();
    opts.delete();

    flex = FlexDelegate();
    opts = InterpreterOptions();
    opts.addDelegate(flex);
    interp = Interpreter.fromFile(_flexModelFile, options: opts);

    // Fresh interpreter should predict ~0 (untrained).
    final inferFresh = interp.getSignatureRunner('infer');
    final predFresh = [
      [0.0],
    ];
    inferFresh.run(
      {
        'x': [
          [1.0],
        ],
      },
      {'output': predFresh},
    );
    inferFresh.close();
    expect(predFresh[0][0], closeTo(0.0, 1e-5));

    // Restore checkpoint A.
    final restoreA = interp.getSignatureRunner('restore');
    final restoreStatusA = [0];
    restoreA.run(
      {
        'checkpoint_path': [ckptA],
      },
      {'status': restoreStatusA},
    );
    restoreA.close();
    expect(restoreStatusA[0], equals(0));

    final inferAfterA = interp.getSignatureRunner('infer');
    final predAfterA = [
      [0.0],
    ];
    inferAfterA.run(
      {
        'x': [
          [1.0],
        ],
      },
      {'output': predAfterA},
    );
    inferAfterA.close();
    expect(predAfterA[0][0], closeTo(predA[0][0], 1e-5));

    // --- Restore checkpoint B → prediction should match B ---
    final restoreB = interp.getSignatureRunner('restore');
    restoreB.run(
      {
        'checkpoint_path': [ckptB],
      },
      {
        'status': [0],
      },
    );
    restoreB.close();

    final inferAfterB = interp.getSignatureRunner('infer');
    final predAfterB = [
      [0.0],
    ];
    inferAfterB.run(
      {
        'x': [
          [1.0],
        ],
      },
      {'output': predAfterB},
    );
    inferAfterB.close();
    expect(predAfterB[0][0], closeTo(predB[0][0], 1e-5));

    // Intentionally leak interp, flex, opts, and tmpDir. Closing or
    // deleting them can trigger SIGBUS from TF background threads.
    // The OS reclaims everything when the process exits.
  });
}
