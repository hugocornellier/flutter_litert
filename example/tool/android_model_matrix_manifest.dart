// Shared, pure-Dart manifest for the Android device matrix and its host-side
// result merger. Keep the published repository revisions pinned so a later
// model update cannot silently change a benchmark run.

const androidMatrixShardCount = 5;

/// Shard index meaning "every model in one execution".
///
/// The five bins exist to fit Firebase Test Lab's 45-minute physical-device
/// limit, and each bin costs one of five daily physical runs. A narrowed mode
/// set (for example dropping the Qualcomm NPU modes on a non-Qualcomm SoC)
/// finishes all 29 models well inside that limit, so collapsing to a single
/// execution buys device coverage with the quota the extra shards would spend.
const androidMatrixAllShards = -1;

const androidMatrixRepositoryCommits = <String, String>{
  'face_detection_tflite': '0a2a60ab69afbfbeefc2d31c0324f8e316935c58',
  'pose_detection': 'fff3b69a4cd7ece1b3d08de39da9d69ff0480c72',
  'hand_detection': '20ece614dbaa65e457a132016570e2d34b6fa6d7',
  'animal_detection': '3f8c5e866c5f5cb8ce995bf77ef51059a2247cbb',
  'cat_detection': '500d6ae9b1c102cd199cff42ac3419016ebc79cc',
  'dog_detection': '98fde417792bd1345734fcfd3fb842955991174c',
  'object_detection': '595bd929824b7b56a8a34bf2248cdd1f173fa2dd',
};

class AndroidMatrixModel {
  const AndroidMatrixModel(this.repository, this.fileName, this.shard);

  final String repository;
  final String fileName;
  final int shard;

  String get name => fileName.substring(0, fileName.length - '.tflite'.length);
  String get label => '$repository/$name';
  String get repositoryCommit => androidMatrixRepositoryCommits[repository]!;
  String get sourceRelativePath => '$repository/assets/models/$fileName';
  String get assetPath => 'assets/models/model_matrix/${repository}__$fileName';

  Map<String, Object?> toJson() => {
    'repository': repository,
    'repository_commit': repositoryCommit,
    'model_name': name,
    'model_file': fileName,
    'shard': shard,
    'asset_path': assetPath,
  };
}

// These five bins are balanced from the macOS CPU-reference sweep. The known
// native-crash probe (full-range sparse) is deliberately last in its shard so
// a process-level abort cannot erase completed rows for unrelated models.
const androidMatrixModels = <AndroidMatrixModel>[
  AndroidMatrixModel('pose_detection', 'yolov8n_float32.tflite', 0),
  AndroidMatrixModel('face_detection_tflite', 'mobilefacenet.tflite', 0),
  AndroidMatrixModel('hand_detection', 'hand_detection.tflite', 0),
  AndroidMatrixModel('face_detection_tflite', 'face_landmark.tflite', 0),
  AndroidMatrixModel(
    'face_detection_tflite',
    'selfie_segmenter_landscape.tflite',
    0,
  ),

  AndroidMatrixModel('object_detection', 'efficientdet_lite2.tflite', 1),
  AndroidMatrixModel('object_detection', 'efficientdet_lite0.tflite', 1),
  AndroidMatrixModel('pose_detection', 'pose_landmark_lite.tflite', 1),
  AndroidMatrixModel('hand_detection', 'hand_landmark_full.tflite', 1),
  AndroidMatrixModel('face_detection_tflite', 'face_detection_front.tflite', 1),
  AndroidMatrixModel('hand_detection', 'gesture_embedder.tflite', 1),

  AndroidMatrixModel('face_detection_tflite', 'selfie_multiclass.tflite', 2),
  AndroidMatrixModel(
    'animal_detection',
    'superanimal_ssdlite_float16.tflite',
    2,
  ),
  AndroidMatrixModel('pose_detection', 'pose_landmark_full.tflite', 2),
  AndroidMatrixModel(
    'animal_detection',
    'species_classifier_float16.tflite',
    2,
  ),
  AndroidMatrixModel('face_detection_tflite', 'selfie_segmenter.tflite', 2),

  AndroidMatrixModel('dog_detection', 'dog_face_landmarks_full.tflite', 3),
  AndroidMatrixModel(
    'animal_detection',
    'superanimal_rtmpose_s_float16.tflite',
    3,
  ),
  AndroidMatrixModel('cat_detection', 'cat_face_localizer.tflite', 3),
  AndroidMatrixModel(
    'face_detection_tflite',
    'face_detection_full_range.tflite',
    3,
  ),
  AndroidMatrixModel('face_detection_tflite', 'face_blendshapes.tflite', 3),
  AndroidMatrixModel('hand_detection', 'canned_gesture_classifier.tflite', 3),
  AndroidMatrixModel(
    'face_detection_tflite',
    'face_detection_full_range_sparse.tflite',
    3,
  ),

  AndroidMatrixModel('cat_detection', 'cat_face_landmarks_full.tflite', 4),
  AndroidMatrixModel('pose_detection', 'pose_landmark_heavy.tflite', 4),
  AndroidMatrixModel('dog_detection', 'dog_face_localizer.tflite', 4),
  AndroidMatrixModel('face_detection_tflite', 'face_detection_back.tflite', 4),
  AndroidMatrixModel('face_detection_tflite', 'iris_landmark.tflite', 4),
  AndroidMatrixModel(
    'face_detection_tflite',
    'face_detection_short_range.tflite',
    4,
  ),
];

class AndroidMatrixMode {
  const AndroidMatrixMode({
    required this.engine,
    required this.label,
    required this.delegate,
    required this.accelerators,
    required this.precision,
    required this.asyncSupported,
  });

  final String engine;
  final String label;
  final String? delegate;
  final String accelerators;
  final String precision;
  final bool asyncSupported;

  Map<String, Object?> toJson() => {
    'engine': engine,
    'mode': label,
    'delegate': delegate,
    'accelerators': accelerators,
    'precision': precision,
    'async_supported': asyncSupported,
  };
}

// Interpreter exposes CPU, XNNPACK, Flex, and Android's GL/CL GPU delegate.
// CompiledModel exposes three accelerators; every non-empty accelerator set is
// represented, plus the GPU-only fp16 precision variant used in production.
const androidMatrixModes = <AndroidMatrixMode>[
  AndroidMatrixMode(
    engine: 'interpreter',
    label: 'interpreter_cpu_4t',
    delegate: 'none',
    accelerators: 'cpu',
    precision: 'fp32',
    asyncSupported: false,
  ),
  AndroidMatrixMode(
    engine: 'interpreter',
    label: 'interpreter_xnnpack_4t',
    delegate: 'xnnpack',
    accelerators: 'cpu_xnnpack',
    precision: 'fp32',
    asyncSupported: false,
  ),
  AndroidMatrixMode(
    engine: 'interpreter',
    label: 'interpreter_flex',
    delegate: 'flex_select_tf_ops',
    accelerators: 'cpu_flex',
    precision: 'fp32',
    asyncSupported: false,
  ),
  AndroidMatrixMode(
    engine: 'interpreter',
    label: 'interpreter_gpu_v2_gl_cl_fp16',
    delegate: 'gpu_v2_gl_cl',
    accelerators: 'gpu',
    precision: 'fp16',
    asyncSupported: false,
  ),
  AndroidMatrixMode(
    engine: 'interpreter',
    label: 'interpreter_gpu_v2_gl_cl_fp32',
    delegate: 'gpu_v2_gl_cl',
    accelerators: 'gpu',
    precision: 'fp32',
    asyncSupported: false,
  ),
  AndroidMatrixMode(
    engine: 'compiled_model',
    label: 'compiled_cpu_fp32',
    delegate: null,
    accelerators: 'cpu',
    precision: 'fp32',
    asyncSupported: true,
  ),
  AndroidMatrixMode(
    engine: 'compiled_model',
    label: 'compiled_npu_fp32',
    delegate: null,
    accelerators: 'npu',
    precision: 'fp32',
    asyncSupported: false,
  ),
  AndroidMatrixMode(
    engine: 'compiled_model',
    label: 'compiled_npu_cpu_fp32',
    delegate: null,
    accelerators: 'npu+cpu',
    precision: 'fp32',
    asyncSupported: false,
  ),
  AndroidMatrixMode(
    engine: 'compiled_model',
    label: 'compiled_npu_gpu_cpu_fp32',
    delegate: null,
    accelerators: 'npu+gpu+cpu',
    precision: 'fp32',
    asyncSupported: false,
  ),
  AndroidMatrixMode(
    engine: 'compiled_model',
    label: 'compiled_npu_gpu_fp32',
    delegate: null,
    accelerators: 'npu+gpu',
    precision: 'fp32',
    asyncSupported: false,
  ),
  AndroidMatrixMode(
    engine: 'compiled_model',
    label: 'compiled_gpu_cpu_fp32',
    delegate: null,
    accelerators: 'gpu+cpu',
    precision: 'fp32',
    asyncSupported: false,
  ),
  AndroidMatrixMode(
    engine: 'compiled_model',
    label: 'compiled_gpu_fp16',
    delegate: null,
    accelerators: 'gpu',
    precision: 'fp16',
    asyncSupported: false,
  ),
  AndroidMatrixMode(
    engine: 'compiled_model',
    label: 'compiled_gpu_fp32',
    delegate: null,
    accelerators: 'gpu',
    precision: 'fp32',
    asyncSupported: false,
  ),
];

Iterable<AndroidMatrixModel> androidMatrixModelsForShard(int shard) =>
    shard == androidMatrixAllShards
    ? androidMatrixModels
    : androidMatrixModels.where((model) => model.shard == shard);

Iterable<AndroidMatrixMode> get androidInterpreterMatrixModes =>
    androidMatrixModes.where((mode) => mode.engine == 'interpreter');

Iterable<AndroidMatrixMode> get androidCompiledModelMatrixModes =>
    androidMatrixModes.where((mode) => mode.engine == 'compiled_model');
