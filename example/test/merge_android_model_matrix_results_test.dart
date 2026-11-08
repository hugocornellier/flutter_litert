import 'package:flutter_test/flutter_test.dart';

import '../tool/merge_android_model_matrix_results.dart'
    show matrixReportSource;

Map<String, Object?> _shard(int index, String manufacturer, String model) => {
  'matrix_shard': index,
  'device_manufacturer': manufacturer,
  'device_model': model,
};

void main() {
  test('names the device recorded in the shard metadata', () {
    expect(
      matrixReportSource([
        for (var shard = 0; shard < 5; shard++)
          _shard(shard, 'samsung', 'SM-S911U'),
      ]),
      '5 Firebase Test Lab samsung SM-S911U shards.',
    );
  });

  test('does not repeat a manufacturer the model already names', () {
    expect(
      matrixReportSource([_shard(-1, 'Google', 'Google Pixel 9 Pro')]),
      '1 Firebase Test Lab Google Pixel 9 Pro shard.',
    );
  });

  test('lists each distinct device once', () {
    expect(
      matrixReportSource([
        _shard(0, 'samsung', 'SM-S911U'),
        _shard(1, 'samsung', 'SM-A566B'),
        _shard(2, 'samsung', 'SM-S911U'),
      ]),
      '3 Firebase Test Lab samsung SM-S911U, samsung SM-A566B shards.',
    );
  });

  test('falls back to Test Lab when no shard recorded a device', () {
    expect(
      matrixReportSource([
        {'matrix_shard': 0, 'device_model': 'unknown'},
        {'matrix_shard': 1},
      ]),
      '2 Firebase Test Lab shards.',
    );
    expect(matrixReportSource(const []), 'Firebase Test Lab shards.');
  });
}
