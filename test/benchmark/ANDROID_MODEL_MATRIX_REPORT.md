# Android physical-device model matrix

Generated 2026-08-02T16:02:57.977727Z from five Firebase Test Lab Galaxy S23 shards.

- Rows: 377/377 (rectangular: true)
- Status: `{"ok":321,"error":48,"unsupported":8}`
- Accuracy failures: 80
- Native crashes: 0
- Synthesized rows: 0

Each successful cell is `OK p50 ms` or `ACC FAIL p50 ms`. Timing is synchronous invocation with managed I/O for CompiledModel and invoke-only for Interpreter.

| model | interpreter_cpu_4t | interpreter_xnnpack_4t | interpreter_flex | interpreter_gpu_v2_gl_cl_fp16 | interpreter_gpu_v2_gl_cl_fp32 | compiled_cpu_fp32 | compiled_npu_fp32 | compiled_npu_cpu_fp32 | compiled_npu_gpu_cpu_fp32 | compiled_npu_gpu_fp32 | compiled_gpu_cpu_fp32 | compiled_gpu_fp16 | compiled_gpu_fp32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pose_detection/yolov8n_float32 | OK 112.6 ms | OK 68.4 ms | OK 113.1 ms | ACC FAIL 17.4 ms | OK 30.0 ms | OK 157.3 ms | ACC FAIL 20.2 ms | ACC FAIL 19.3 ms | ACC FAIL 19.4 ms | ACC FAIL 18.9 ms | OK 18.9 ms | ACC FAIL 12.4 ms | OK 20.3 ms |
| face_detection_tflite/mobilefacenet | OK 15.3 ms | OK 3.73 ms | OK 15.3 ms | OK 2.38 ms | OK 2.90 ms | OK 8.32 ms | OK 6.01 ms | OK 5.89 ms | OK 5.74 ms | OK 5.81 ms | OK 2.39 ms | ERROR 504 | ERROR 504 |
| hand_detection/hand_detection | OK 17.0 ms | OK 5.97 ms | OK 22.1 ms | ACC FAIL 3.06 ms | OK 4.73 ms | OK 14.2 ms | OK 4.75 ms | OK 4.62 ms | OK 4.60 ms | OK 4.81 ms | OK 3.71 ms | ACC FAIL 2.18 ms | OK 3.48 ms |
| face_detection_tflite/face_landmark | OK 3.23 ms | OK 1.19 ms | OK 2.53 ms | ACC FAIL 1.28 ms | OK 1.51 ms | OK 1.70 ms | OK 3.10 ms | OK 3.25 ms | OK 3.29 ms | OK 2.96 ms | OK 1.42 ms | OK 0.98 ms | OK 1.66 ms |
| face_detection_tflite/selfie_segmenter_landscape | OK 10.8 ms | OK 0.92 ms | OK 10.9 ms | ACC FAIL 1.49 ms | OK 2.07 ms | OK 1.74 ms | ERROR 504 | OK 1.70 ms | OK 1.39 ms | OK 1.39 ms | OK 1.40 ms | ACC FAIL 1.22 ms | OK 1.72 ms |
| object_detection/efficientdet_lite2 | OK 117.1 ms | OK 59.6 ms | OK 154.7 ms | ACC FAIL 20.0 ms | OK 30.6 ms | OK 126.5 ms | ACC FAIL 37.8 ms | ACC FAIL 40.9 ms | ACC FAIL 32.2 ms | ACC FAIL 35.8 ms | OK 30.0 ms | ACC FAIL 21.6 ms | OK 31.9 ms |
| object_detection/efficientdet_lite0 | OK 40.5 ms | OK 16.5 ms | OK 55.7 ms | ACC FAIL 9.14 ms | OK 12.7 ms | OK 39.4 ms | ACC FAIL 14.9 ms | ACC FAIL 14.3 ms | ACC FAIL 15.0 ms | ACC FAIL 13.2 ms | OK 15.2 ms | ACC FAIL 8.68 ms | OK 13.2 ms |
| pose_detection/pose_landmark_lite | OK 15.8 ms | OK 4.19 ms | OK 20.0 ms | ACC FAIL 4.04 ms | OK 5.80 ms | OK 10.5 ms | ACC FAIL 8.41 ms | ACC FAIL 8.04 ms | ACC FAIL 8.28 ms | ACC FAIL 8.52 ms | OK 5.17 ms | ACC FAIL 3.47 ms | OK 4.93 ms |
| hand_detection/hand_landmark_full | OK 13.1 ms | OK 5.01 ms | OK 17.9 ms | ACC FAIL 3.98 ms | OK 5.55 ms | OK 12.0 ms | ACC FAIL 3.99 ms | ACC FAIL 3.92 ms | ACC FAIL 4.15 ms | ACC FAIL 4.26 ms | OK 3.61 ms | ACC FAIL 2.98 ms | OK 4.14 ms |
| face_detection_tflite/face_detection_front | OK 3.06 ms | OK 0.75 ms | OK 1.91 ms | ACC FAIL 0.87 ms | OK 1.33 ms | OK 1.51 ms | OK 2.80 ms | OK 3.07 ms | OK 2.84 ms | OK 3.10 ms | OK 1.14 ms | ACC FAIL 1.03 ms | OK 1.17 ms |
| hand_detection/gesture_embedder | OK 0.036 ms | OK 0.023 ms | OK 0.029 ms | ACC FAIL 0.55 ms | OK 0.59 ms | OK 0.022 ms | ERROR 3 | ERROR 3 | ERROR 3 | ERROR 3 | ERROR 3 | ERROR 3 | ERROR 3 |
| face_detection_tflite/selfie_multiclass | OK 104.7 ms | OK 69.8 ms | OK 115.6 ms | ACC FAIL 12.7 ms | OK 21.9 ms | OK 154.4 ms | ACC FAIL 13.1 ms | ACC FAIL 13.9 ms | ACC FAIL 13.3 ms | ACC FAIL 13.3 ms | OK 16.0 ms | ACC FAIL 9.90 ms | OK 16.4 ms |
| animal_detection/superanimal_ssdlite_float16 | OK 28.3 ms | OK 7.88 ms | OK 32.3 ms | OK 5.20 ms | OK 7.09 ms | OK 17.4 ms | OK 7.36 ms | OK 7.14 ms | OK 7.60 ms | OK 7.49 ms | OK 17.6 ms | ERROR 504 | ERROR 504 |
| pose_detection/pose_landmark_full | OK 26.4 ms | OK 7.32 ms | OK 34.1 ms | ACC FAIL 5.70 ms | OK 7.96 ms | OK 16.9 ms | ACC FAIL 9.08 ms | ACC FAIL 9.35 ms | ACC FAIL 9.25 ms | ACC FAIL 8.93 ms | OK 7.39 ms | ACC FAIL 4.80 ms | OK 6.98 ms |
| animal_detection/species_classifier_float16 | OK 4.33 ms | OK 1.72 ms | OK 4.42 ms | ACC FAIL 2.23 ms | OK 3.13 ms | OK 3.00 ms | ACC FAIL 4.23 ms | ACC FAIL 4.45 ms | ACC FAIL 4.32 ms | ACC FAIL 4.11 ms | ACC FAIL 3.20 ms | ERROR 504 | ERROR 504 |
| face_detection_tflite/selfie_segmenter | OK 21.0 ms | OK 1.56 ms | OK 19.9 ms | ACC FAIL 1.60 ms | OK 2.39 ms | OK 3.30 ms | ERROR 504 | OK 3.38 ms | OK 2.39 ms | OK 1.82 ms | OK 2.20 ms | ACC FAIL 1.67 ms | OK 2.02 ms |
| dog_detection/dog_face_landmarks_full | OK 187.1 ms | OK 151.6 ms | OK 185.0 ms | UNSUPPORTED | UNSUPPORTED | ERROR 3 | ERROR 504 | OK 169.3 ms | OK 169.0 ms | ERROR 504 | ERROR 3 | ERROR 3 | ERROR 3 |
| animal_detection/superanimal_rtmpose_s_float16 | OK 38.6 ms | OK 13.2 ms | OK 34.7 ms | ACC FAIL 9.28 ms | ACC FAIL 12.1 ms | OK 33.7 ms | ACC FAIL 4.81 ms | ACC FAIL 4.90 ms | ACC FAIL 4.78 ms | ACC FAIL 4.92 ms | ACC FAIL 33.8 ms | ERROR 504 | ERROR 504 |
| cat_detection/cat_face_localizer | OK 50.2 ms | OK 15.3 ms | OK 54.6 ms | UNSUPPORTED | UNSUPPORTED | OK 37.1 ms | ERROR 504 | ACC FAIL 159.1 ms | ACC FAIL 157.4 ms | ERROR 504 | ERROR 3 | ERROR 3 | ERROR 3 |
| face_detection_tflite/face_detection_full_range | OK 7.88 ms | OK 1.97 ms | OK 6.39 ms | OK 2.26 ms | OK 3.12 ms | OK 4.63 ms | OK 3.74 ms | OK 3.77 ms | OK 3.80 ms | OK 3.91 ms | OK 2.65 ms | OK 1.77 ms | OK 2.24 ms |
| face_detection_tflite/face_blendshapes | OK 1.68 ms | OK 0.81 ms | OK 1.52 ms | OK 1.23 ms | OK 1.64 ms | OK 1.00 ms | OK 4.15 ms | OK 4.07 ms | OK 4.09 ms | OK 4.07 ms | ERROR 3 | ERROR 504 | ERROR 3 |
| hand_detection/canned_gesture_classifier | OK 0.001 ms | OK 0.000 ms | OK 0.001 ms | OK 0.29 ms | OK 0.39 ms | OK 0.005 ms | OK 2.68 ms | OK 2.69 ms | OK 2.56 ms | OK 2.78 ms | OK 0.33 ms | OK 0.32 ms | OK 0.27 ms |
| face_detection_tflite/face_detection_full_range_sparse | OK 7.26 ms | OK 1.89 ms | OK 7.84 ms | OK 2.35 ms | OK 3.29 ms | OK 3.90 ms | ERROR 3 | ERROR 3 | ERROR 3 | ERROR 3 | ACC FAIL 5.38 ms | ERROR 504 | ERROR 504 |
| cat_detection/cat_face_landmarks_full | OK 191.8 ms | OK 152.0 ms | OK 182.7 ms | UNSUPPORTED | UNSUPPORTED | ERROR 3 | ERROR 504 | OK 169.7 ms | OK 169.0 ms | ERROR 504 | ERROR 3 | ERROR 3 | ERROR 3 |
| pose_detection/pose_landmark_heavy | OK 68.3 ms | OK 31.7 ms | OK 96.7 ms | ACC FAIL 13.3 ms | OK 19.7 ms | OK 73.1 ms | ACC FAIL 11.3 ms | ACC FAIL 12.4 ms | ACC FAIL 12.3 ms | ACC FAIL 12.1 ms | OK 14.5 ms | ACC FAIL 8.69 ms | OK 14.5 ms |
| dog_detection/dog_face_localizer | OK 48.7 ms | OK 17.4 ms | OK 50.6 ms | UNSUPPORTED | UNSUPPORTED | OK 41.8 ms | ERROR 504 | ACC FAIL 158.7 ms | ACC FAIL 159.0 ms | ERROR 504 | ERROR 3 | ERROR 3 | ERROR 3 |
| face_detection_tflite/face_detection_back | OK 15.6 ms | OK 4.11 ms | OK 17.8 ms | ACC FAIL 2.56 ms | OK 3.79 ms | OK 9.85 ms | OK 4.58 ms | OK 4.63 ms | OK 4.54 ms | OK 4.33 ms | OK 2.95 ms | ACC FAIL 1.69 ms | OK 2.96 ms |
| face_detection_tflite/iris_landmark | OK 2.53 ms | OK 0.97 ms | OK 3.31 ms | OK 2.40 ms | OK 2.52 ms | OK 1.86 ms | OK 3.06 ms | OK 2.94 ms | OK 3.04 ms | OK 3.06 ms | OK 2.44 ms | OK 2.22 ms | OK 2.66 ms |
| face_detection_tflite/face_detection_short_range | OK 2.75 ms | OK 0.91 ms | OK 2.23 ms | ACC FAIL 1.05 ms | OK 1.28 ms | OK 1.67 ms | OK 3.01 ms | OK 3.05 ms | OK 2.86 ms | OK 3.02 ms | OK 0.88 ms | ACC FAIL 1.04 ms | OK 1.19 ms |

## Device metadata

```json
[
  {
    "schema_version": 1,
    "timestamp_utc": "2026-08-02T15:39:11.373722Z",
    "platform": "android",
    "platform_version": "AP3A.240905.015.A2.S911U1UES6DYI3",
    "abi": "android_arm64",
    "device_model": "SM-S911U1",
    "device_manufacturer": "samsung",
    "device_hardware": "qcom",
    "device_board": "kalama",
    "physical_device": true,
    "physical_device_required": true,
    "device_extra": "SDK 35; ABIs=arm64-v8a,armeabi-v7a,armeabi",
    "logical_processors": 8,
    "build_mode": "debug",
    "flutter_litert_commit": "e0c8103e003389b1cad312db3fbc5ca8951bf493",
    "interpreter_runtime_version": "2.22.0-dev0+selfbuilt",
    "matrix_shard": 0,
    "matrix_shard_count": 5,
    "repository_commits": {
      "face_detection_tflite": "0a2a60ab69afbfbeefc2d31c0324f8e316935c58",
      "pose_detection": "fff3b69a4cd7ece1b3d08de39da9d69ff0480c72",
      "hand_detection": "20ece614dbaa65e457a132016570e2d34b6fa6d7",
      "animal_detection": "3f8c5e866c5f5cb8ce995bf77ef51059a2247cbb",
      "cat_detection": "500d6ae9b1c102cd199cff42ac3419016ebc79cc",
      "dog_detection": "98fde417792bd1345734fcfd3fb842955991174c",
      "object_detection": "595bd929824b7b56a8a34bf2248cdd1f173fa2dd"
    },
    "model_count": 5,
    "interpreter_mode_count": 5,
    "compiled_model_mode_count": 8,
    "expected_rows": 65,
    "model_filter": "",
    "mode_filter": "",
    "iterations": 15,
    "warmup": 5,
    "accuracy_kind": "cpu_reference_tensor_parity",
    "accuracy_fixture_candidates": [
      "constant_0_5",
      "ramp_0_05_0_95",
      "scrambled_0_1_0_9",
      "reverse_ramp_0_05_0_95",
      "scrambled_0_2_0_8"
    ],
    "accuracy_fixtures_per_model": 3,
    "absolute_tolerance": 0.0001,
    "relative_tolerance": 0.01,
    "accuracy_enforced": false,
    "compiled_async_policy": "CPU only; mobile accelerator dispatch is kept synchronous because Android GL/CL drivers are thread-affine and runAsync is unvalidated."
  },
  {
    "schema_version": 1,
    "timestamp_utc": "2026-08-02T15:44:01.767057Z",
    "platform": "android",
    "platform_version": "AP3A.240905.015.A2.S911U1UES6DYI3",
    "abi": "android_arm64",
    "device_model": "SM-S911U1",
    "device_manufacturer": "samsung",
    "device_hardware": "qcom",
    "device_board": "kalama",
    "physical_device": true,
    "physical_device_required": true,
    "device_extra": "SDK 35; ABIs=arm64-v8a,armeabi-v7a,armeabi",
    "logical_processors": 8,
    "build_mode": "debug",
    "flutter_litert_commit": "e0c8103e003389b1cad312db3fbc5ca8951bf493",
    "interpreter_runtime_version": "2.22.0-dev0+selfbuilt",
    "matrix_shard": 1,
    "matrix_shard_count": 5,
    "repository_commits": {
      "face_detection_tflite": "0a2a60ab69afbfbeefc2d31c0324f8e316935c58",
      "pose_detection": "fff3b69a4cd7ece1b3d08de39da9d69ff0480c72",
      "hand_detection": "20ece614dbaa65e457a132016570e2d34b6fa6d7",
      "animal_detection": "3f8c5e866c5f5cb8ce995bf77ef51059a2247cbb",
      "cat_detection": "500d6ae9b1c102cd199cff42ac3419016ebc79cc",
      "dog_detection": "98fde417792bd1345734fcfd3fb842955991174c",
      "object_detection": "595bd929824b7b56a8a34bf2248cdd1f173fa2dd"
    },
    "model_count": 6,
    "interpreter_mode_count": 5,
    "compiled_model_mode_count": 8,
    "expected_rows": 78,
    "model_filter": "",
    "mode_filter": "",
    "iterations": 15,
    "warmup": 5,
    "accuracy_kind": "cpu_reference_tensor_parity",
    "accuracy_fixture_candidates": [
      "constant_0_5",
      "ramp_0_05_0_95",
      "scrambled_0_1_0_9",
      "reverse_ramp_0_05_0_95",
      "scrambled_0_2_0_8"
    ],
    "accuracy_fixtures_per_model": 3,
    "absolute_tolerance": 0.0001,
    "relative_tolerance": 0.01,
    "accuracy_enforced": false,
    "compiled_async_policy": "CPU only; mobile accelerator dispatch is kept synchronous because Android GL/CL drivers are thread-affine and runAsync is unvalidated."
  },
  {
    "schema_version": 1,
    "timestamp_utc": "2026-08-02T15:48:03.820871Z",
    "platform": "android",
    "platform_version": "AP3A.240905.015.A2.S911U1UES6DYI3",
    "abi": "android_arm64",
    "device_model": "SM-S911U1",
    "device_manufacturer": "samsung",
    "device_hardware": "qcom",
    "device_board": "kalama",
    "physical_device": true,
    "physical_device_required": true,
    "device_extra": "SDK 35; ABIs=arm64-v8a,armeabi-v7a,armeabi",
    "logical_processors": 8,
    "build_mode": "debug",
    "flutter_litert_commit": "e0c8103e003389b1cad312db3fbc5ca8951bf493",
    "interpreter_runtime_version": "2.22.0-dev0+selfbuilt",
    "matrix_shard": 2,
    "matrix_shard_count": 5,
    "repository_commits": {
      "face_detection_tflite": "0a2a60ab69afbfbeefc2d31c0324f8e316935c58",
      "pose_detection": "fff3b69a4cd7ece1b3d08de39da9d69ff0480c72",
      "hand_detection": "20ece614dbaa65e457a132016570e2d34b6fa6d7",
      "animal_detection": "3f8c5e866c5f5cb8ce995bf77ef51059a2247cbb",
      "cat_detection": "500d6ae9b1c102cd199cff42ac3419016ebc79cc",
      "dog_detection": "98fde417792bd1345734fcfd3fb842955991174c",
      "object_detection": "595bd929824b7b56a8a34bf2248cdd1f173fa2dd"
    },
    "model_count": 5,
    "interpreter_mode_count": 5,
    "compiled_model_mode_count": 8,
    "expected_rows": 65,
    "model_filter": "",
    "mode_filter": "",
    "iterations": 15,
    "warmup": 5,
    "accuracy_kind": "cpu_reference_tensor_parity",
    "accuracy_fixture_candidates": [
      "constant_0_5",
      "ramp_0_05_0_95",
      "scrambled_0_1_0_9",
      "reverse_ramp_0_05_0_95",
      "scrambled_0_2_0_8"
    ],
    "accuracy_fixtures_per_model": 3,
    "absolute_tolerance": 0.0001,
    "relative_tolerance": 0.01,
    "accuracy_enforced": false,
    "compiled_async_policy": "CPU only; mobile accelerator dispatch is kept synchronous because Android GL/CL drivers are thread-affine and runAsync is unvalidated."
  },
  {
    "schema_version": 1,
    "timestamp_utc": "2026-08-02T15:54:26.306701Z",
    "platform": "android",
    "platform_version": "AP3A.240905.015.A2.S911U1UES6DYI3",
    "abi": "android_arm64",
    "device_model": "SM-S911U1",
    "device_manufacturer": "samsung",
    "device_hardware": "qcom",
    "device_board": "kalama",
    "physical_device": true,
    "physical_device_required": true,
    "device_extra": "SDK 35; ABIs=arm64-v8a,armeabi-v7a,armeabi",
    "logical_processors": 8,
    "build_mode": "debug",
    "flutter_litert_commit": "e0c8103e003389b1cad312db3fbc5ca8951bf493",
    "interpreter_runtime_version": "2.22.0-dev0+selfbuilt",
    "matrix_shard": 3,
    "matrix_shard_count": 5,
    "repository_commits": {
      "face_detection_tflite": "0a2a60ab69afbfbeefc2d31c0324f8e316935c58",
      "pose_detection": "fff3b69a4cd7ece1b3d08de39da9d69ff0480c72",
      "hand_detection": "20ece614dbaa65e457a132016570e2d34b6fa6d7",
      "animal_detection": "3f8c5e866c5f5cb8ce995bf77ef51059a2247cbb",
      "cat_detection": "500d6ae9b1c102cd199cff42ac3419016ebc79cc",
      "dog_detection": "98fde417792bd1345734fcfd3fb842955991174c",
      "object_detection": "595bd929824b7b56a8a34bf2248cdd1f173fa2dd"
    },
    "model_count": 7,
    "interpreter_mode_count": 5,
    "compiled_model_mode_count": 8,
    "expected_rows": 91,
    "model_filter": "",
    "mode_filter": "",
    "iterations": 15,
    "warmup": 5,
    "accuracy_kind": "cpu_reference_tensor_parity",
    "accuracy_fixture_candidates": [
      "constant_0_5",
      "ramp_0_05_0_95",
      "scrambled_0_1_0_9",
      "reverse_ramp_0_05_0_95",
      "scrambled_0_2_0_8"
    ],
    "accuracy_fixtures_per_model": 3,
    "absolute_tolerance": 0.0001,
    "relative_tolerance": 0.01,
    "accuracy_enforced": false,
    "compiled_async_policy": "CPU only; mobile accelerator dispatch is kept synchronous because Android GL/CL drivers are thread-affine and runAsync is unvalidated."
  },
  {
    "schema_version": 1,
    "timestamp_utc": "2026-08-02T16:01:19.093913Z",
    "platform": "android",
    "platform_version": "AP3A.240905.015.A2.S911U1UES6DYI3",
    "abi": "android_arm64",
    "device_model": "SM-S911U1",
    "device_manufacturer": "samsung",
    "device_hardware": "qcom",
    "device_board": "kalama",
    "physical_device": true,
    "physical_device_required": true,
    "device_extra": "SDK 35; ABIs=arm64-v8a,armeabi-v7a,armeabi",
    "logical_processors": 8,
    "build_mode": "debug",
    "flutter_litert_commit": "e0c8103e003389b1cad312db3fbc5ca8951bf493",
    "interpreter_runtime_version": "2.22.0-dev0+selfbuilt",
    "matrix_shard": 4,
    "matrix_shard_count": 5,
    "repository_commits": {
      "face_detection_tflite": "0a2a60ab69afbfbeefc2d31c0324f8e316935c58",
      "pose_detection": "fff3b69a4cd7ece1b3d08de39da9d69ff0480c72",
      "hand_detection": "20ece614dbaa65e457a132016570e2d34b6fa6d7",
      "animal_detection": "3f8c5e866c5f5cb8ce995bf77ef51059a2247cbb",
      "cat_detection": "500d6ae9b1c102cd199cff42ac3419016ebc79cc",
      "dog_detection": "98fde417792bd1345734fcfd3fb842955991174c",
      "object_detection": "595bd929824b7b56a8a34bf2248cdd1f173fa2dd"
    },
    "model_count": 6,
    "interpreter_mode_count": 5,
    "compiled_model_mode_count": 8,
    "expected_rows": 78,
    "model_filter": "",
    "mode_filter": "",
    "iterations": 15,
    "warmup": 5,
    "accuracy_kind": "cpu_reference_tensor_parity",
    "accuracy_fixture_candidates": [
      "constant_0_5",
      "ramp_0_05_0_95",
      "scrambled_0_1_0_9",
      "reverse_ramp_0_05_0_95",
      "scrambled_0_2_0_8"
    ],
    "accuracy_fixtures_per_model": 3,
    "absolute_tolerance": 0.0001,
    "relative_tolerance": 0.01,
    "accuracy_enforced": false,
    "compiled_async_policy": "CPU only; mobile accelerator dispatch is kept synchronous because Android GL/CL drivers are thread-affine and runAsync is unvalidated."
  }
]
```
