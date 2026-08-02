import 'package:flutter_test/flutter_test.dart';

import '../tool/android_model_matrix_manifest.dart';

void main() {
  test('pins 29 unique models into five balanced shards', () {
    expect(androidMatrixModels, hasLength(29));
    expect(
      androidMatrixModels.map((model) => model.label).toSet(),
      hasLength(29),
    );
    expect(androidMatrixRepositoryCommits, hasLength(7));
    expect(
      androidMatrixRepositoryCommits.values,
      everyElement(matches(RegExp(r'^[0-9a-f]{40}$'))),
    );
    expect(
      {
        for (var shard = 0; shard < androidMatrixShardCount; shard++)
          shard: androidMatrixModelsForShard(shard).length,
      },
      {0: 5, 1: 6, 2: 5, 3: 7, 4: 6},
    );
    expect(
      androidMatrixModels.map((model) => model.assetPath),
      everyElement(startsWith('assets/models/model_matrix/')),
    );
    expect(
      androidMatrixModels.map((model) => model.assetPath.split('/').length),
      everyElement(4),
    );
  });

  test('the all-shards sentinel selects every model exactly once', () {
    final all = androidMatrixModelsForShard(androidMatrixAllShards).toList();
    expect(all, hasLength(29));
    expect(all.map((model) => model.label).toSet(), hasLength(29));
    expect(
      all.map((model) => model.label).toSet(),
      androidMatrixModels.map((model) => model.label).toSet(),
    );
    // The sentinel must not collide with a real bin, or a single-execution run
    // would silently collect a subset and still look rectangular.
    expect(androidMatrixAllShards, lessThan(0));
    expect(
      androidMatrixModels.map((model) => model.shard),
      everyElement(isNot(androidMatrixAllShards)),
    );
  });

  test('covers both APIs and every CompiledModel accelerator set', () {
    expect(androidInterpreterMatrixModes, hasLength(5));
    expect(androidCompiledModelMatrixModes, hasLength(8));
    expect(androidMatrixModes.map((mode) => mode.label).toSet(), hasLength(13));
    expect(
      androidCompiledModelMatrixModes
          .where((mode) => mode.precision == 'fp32')
          .map((mode) => mode.accelerators)
          .toSet(),
      {'cpu', 'gpu', 'npu', 'gpu+cpu', 'npu+cpu', 'npu+gpu', 'npu+gpu+cpu'},
    );
  });
}
