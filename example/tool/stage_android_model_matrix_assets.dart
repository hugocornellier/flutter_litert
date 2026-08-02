import 'dart:io';

import 'android_model_matrix_manifest.dart';

Never _usage([String? message]) {
  if (message != null) stderr.writeln('Error: $message');
  stderr.writeln(
    'Usage: dart run example/tool/stage_android_model_matrix_assets.dart '
    '--repositories-root <dir> --asset-root <dir> --shard <0-4|all>',
  );
  exit(64);
}

String _requiredOption(List<String> args, String name) {
  final index = args.indexOf(name);
  if (index < 0 || index + 1 >= args.length) _usage('Missing $name.');
  return args[index + 1];
}

Future<void> main(List<String> args) async {
  final repositoriesRoot = Directory(
    _requiredOption(args, '--repositories-root'),
  ).absolute;
  final assetRoot = Directory(_requiredOption(args, '--asset-root')).absolute;
  // Test Lab shards to fit the 45-minute physical-device limit. A tethered
  // macOS or iOS run has no such limit, so `--shard all` stages the complete
  // 29-model set into a single bundle.
  final shardText = _requiredOption(args, '--shard');
  final shard = shardText == 'all' ? null : int.tryParse(shardText);
  if (shardText != 'all' &&
      (shard == null || shard < 0 || shard >= androidMatrixShardCount)) {
    _usage(
      '--shard must be "all" or between 0 and '
      '${androidMatrixShardCount - 1}.',
    );
  }
  if (!assetRoot.path
      .replaceAll('\\', '/')
      .endsWith('/assets/models/model_matrix')) {
    _usage(
      '--asset-root must end in /assets/models/model_matrix; refusing to '
      'replace ${assetRoot.path}.',
    );
  }
  if (!repositoriesRoot.existsSync()) {
    _usage('Repository root does not exist: ${repositoriesRoot.path}.');
  }

  final models = androidMatrixModelsForShard(
    shard ?? androidMatrixAllShards,
  ).toList();
  if (models.isEmpty) _usage('Shard $shardText contains no models.');

  // This is a generated, gitignored asset directory with a deliberately strict
  // suffix guard above. Replacing it prevents a previous shard's models from
  // leaking into the next APK.
  if (assetRoot.existsSync()) assetRoot.deleteSync(recursive: true);
  assetRoot.createSync(recursive: true);

  final checkedRepositories = <String>{};
  var totalBytes = 0;
  for (final model in models) {
    final repositoryDirectory = Directory(
      '${repositoriesRoot.path}/${model.repository}',
    );
    if (checkedRepositories.add(model.repository)) {
      final result = await Process.run('git', [
        'rev-parse',
        'HEAD',
      ], workingDirectory: repositoryDirectory.path);
      if (result.exitCode != 0) {
        throw StateError(
          'Could not resolve ${model.repository}: ${result.stderr}',
        );
      }
      final actualCommit = result.stdout.toString().trim();
      if (actualCommit != model.repositoryCommit) {
        throw StateError(
          '${model.repository} is at $actualCommit; matrix manifest pins '
          '${model.repositoryCommit}.',
        );
      }
    }

    final source = File(
      '${repositoriesRoot.path}/${model.repository}/assets/models/'
      '${model.fileName}',
    );
    if (!source.existsSync()) {
      throw StateError('Pinned model is missing: ${source.path}.');
    }
    final destination = File(
      '${assetRoot.path}/${model.repository}__${model.fileName}',
    );
    source.copySync(destination.path);
    final bytes = destination.lengthSync();
    totalBytes += bytes;
    stdout.writeln(
      'staged shard=$shardText model=${model.label} bytes=$bytes '
      'commit=${model.repositoryCommit}',
    );
  }

  stdout.writeln(
    'staged shard=$shardText models=${models.length} bytes=$totalBytes '
    'asset_root=${assetRoot.path}',
  );
}
