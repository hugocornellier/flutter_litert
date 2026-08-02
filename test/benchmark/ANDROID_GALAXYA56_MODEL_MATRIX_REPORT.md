# Android physical-device model matrix

Generated 2026-08-04T00:53:19.182830Z from five Firebase Test Lab Galaxy S23 shards.

- Rows: 261/261 (rectangular: true)
- Status: `{"ok":223,"error":30,"unsupported":8}`
- Accuracy failures: 35
- Native crashes: 0
- Synthesized rows: 0

Each successful cell is `OK p50 ms` or `ACC FAIL p50 ms`. Timing is synchronous invocation with managed I/O for CompiledModel and invoke-only for Interpreter.

| model | interpreter_cpu_4t | interpreter_xnnpack_4t | interpreter_flex | interpreter_gpu_v2_gl_cl_fp16 | interpreter_gpu_v2_gl_cl_fp32 | compiled_cpu_fp32 | compiled_npu_fp32 | compiled_npu_cpu_fp32 | compiled_npu_gpu_cpu_fp32 | compiled_npu_gpu_fp32 | compiled_gpu_cpu_fp32 | compiled_gpu_fp16 | compiled_gpu_fp32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pose_detection/yolov8n_float32 | OK 161.1 ms | OK 78.6 ms | OK 144.1 ms | ACC FAIL 33.8 ms | OK 52.8 ms | OK 239.0 ms | MISSING | MISSING | MISSING | MISSING | OK 55.9 ms | ACC FAIL 38.8 ms | OK 56.2 ms |
| face_detection_tflite/mobilefacenet | OK 20.4 ms | OK 5.12 ms | OK 19.9 ms | OK 3.20 ms | OK 4.68 ms | OK 14.1 ms | MISSING | MISSING | MISSING | MISSING | OK 5.10 ms | ERROR 504 | ERROR 504 |
| hand_detection/hand_detection | OK 24.9 ms | OK 7.38 ms | OK 29.6 ms | ACC FAIL 7.15 ms | OK 12.1 ms | OK 21.8 ms | MISSING | MISSING | MISSING | MISSING | OK 9.01 ms | ACC FAIL 6.07 ms | OK 9.21 ms |
| face_detection_tflite/face_landmark | OK 4.63 ms | OK 1.15 ms | OK 5.11 ms | ACC FAIL 3.29 ms | OK 4.20 ms | OK 2.92 ms | MISSING | MISSING | MISSING | MISSING | OK 3.23 ms | OK 2.64 ms | OK 3.58 ms |
| face_detection_tflite/selfie_segmenter_landscape | OK 19.5 ms | OK 1.18 ms | OK 20.2 ms | ACC FAIL 3.05 ms | OK 4.08 ms | OK 2.92 ms | MISSING | MISSING | MISSING | MISSING | OK 3.86 ms | ACC FAIL 3.00 ms | OK 4.07 ms |
| object_detection/efficientdet_lite2 | OK 166.2 ms | OK 67.9 ms | OK 199.6 ms | ACC FAIL 51.6 ms | OK 75.8 ms | OK 201.5 ms | MISSING | MISSING | MISSING | MISSING | OK 76.0 ms | ACC FAIL 61.5 ms | OK 76.3 ms |
| object_detection/efficientdet_lite0 | OK 65.2 ms | OK 21.7 ms | OK 69.9 ms | ACC FAIL 25.1 ms | OK 32.0 ms | OK 62.3 ms | MISSING | MISSING | MISSING | MISSING | OK 32.8 ms | ACC FAIL 28.1 ms | OK 33.1 ms |
| pose_detection/pose_landmark_lite | OK 23.9 ms | OK 6.84 ms | OK 22.7 ms | ACC FAIL 9.55 ms | OK 15.7 ms | OK 16.4 ms | MISSING | MISSING | MISSING | MISSING | OK 11.4 ms | ACC FAIL 8.30 ms | OK 10.6 ms |
| hand_detection/hand_landmark_full | OK 22.1 ms | OK 5.99 ms | OK 18.4 ms | ACC FAIL 7.50 ms | OK 11.5 ms | OK 18.6 ms | MISSING | MISSING | MISSING | MISSING | OK 6.81 ms | ACC FAIL 6.37 ms | OK 7.43 ms |
| face_detection_tflite/face_detection_front | OK 3.34 ms | OK 0.86 ms | OK 3.63 ms | ACC FAIL 3.18 ms | OK 2.43 ms | OK 2.32 ms | MISSING | MISSING | MISSING | MISSING | OK 2.66 ms | ACC FAIL 2.21 ms | OK 1.94 ms |
| hand_detection/gesture_embedder | OK 0.028 ms | OK 0.022 ms | OK 0.029 ms | OK 2.15 ms | OK 1.93 ms | OK 0.050 ms | MISSING | MISSING | MISSING | MISSING | ERROR 3 | ERROR 3 | ERROR 3 |
| face_detection_tflite/selfie_multiclass | OK 150.6 ms | OK 70.5 ms | OK 163.9 ms | ACC FAIL 29.9 ms | OK 49.3 ms | OK 214.1 ms | MISSING | MISSING | MISSING | MISSING | OK 51.5 ms | ACC FAIL 30.1 ms | OK 51.7 ms |
| animal_detection/superanimal_ssdlite_float16 | OK 36.1 ms | OK 12.5 ms | OK 41.0 ms | OK 13.4 ms | OK 23.6 ms | OK 27.5 ms | MISSING | MISSING | MISSING | MISSING | OK 29.5 ms | ERROR 504 | ERROR 504 |
| pose_detection/pose_landmark_full | OK 35.2 ms | OK 10.6 ms | OK 33.5 ms | ACC FAIL 12.9 ms | OK 20.9 ms | OK 27.1 ms | MISSING | MISSING | MISSING | MISSING | OK 13.4 ms | ACC FAIL 11.2 ms | OK 15.9 ms |
| animal_detection/species_classifier_float16 | OK 7.11 ms | OK 2.68 ms | OK 9.83 ms | ACC FAIL 2.94 ms | OK 4.52 ms | OK 4.69 ms | MISSING | MISSING | MISSING | MISSING | ACC FAIL 6.72 ms | ERROR 504 | ERROR 504 |
| face_detection_tflite/selfie_segmenter | OK 28.9 ms | OK 2.36 ms | OK 32.9 ms | ACC FAIL 3.54 ms | OK 5.80 ms | OK 5.41 ms | MISSING | MISSING | MISSING | MISSING | OK 5.80 ms | ACC FAIL 3.62 ms | OK 5.70 ms |
| dog_detection/dog_face_landmarks_full | OK 306.8 ms | OK 228.9 ms | OK 302.8 ms | UNSUPPORTED | UNSUPPORTED | ERROR 3 | MISSING | MISSING | MISSING | MISSING | ERROR 3 | ERROR 3 | ERROR 3 |
| animal_detection/superanimal_rtmpose_s_float16 | OK 54.6 ms | OK 15.9 ms | OK 51.3 ms | ACC FAIL 10.5 ms | ACC FAIL 14.6 ms | OK 51.9 ms | MISSING | MISSING | MISSING | MISSING | ACC FAIL 51.4 ms | ERROR 504 | ERROR 504 |
| cat_detection/cat_face_localizer | OK 71.3 ms | OK 21.4 ms | OK 73.9 ms | UNSUPPORTED | UNSUPPORTED | OK 56.3 ms | MISSING | MISSING | MISSING | MISSING | ERROR 3 | ERROR 3 | ERROR 3 |
| face_detection_tflite/face_detection_full_range | OK 14.4 ms | OK 2.65 ms | OK 12.7 ms | OK 3.76 ms | OK 6.76 ms | OK 7.47 ms | MISSING | MISSING | MISSING | MISSING | OK 6.69 ms | OK 3.73 ms | OK 6.93 ms |
| face_detection_tflite/face_blendshapes | OK 5.17 ms | OK 0.89 ms | OK 2.26 ms | ACC FAIL 2.41 ms | OK 2.20 ms | OK 1.79 ms | MISSING | MISSING | MISSING | MISSING | ERROR 3 | ERROR 504 | ERROR 3 |
| hand_detection/canned_gesture_classifier | OK 0.001 ms | OK 0.000 ms | OK 0.001 ms | OK 1.19 ms | OK 0.80 ms | OK 0.003 ms | MISSING | MISSING | MISSING | MISSING | OK 1.03 ms | OK 1.25 ms | OK 1.04 ms |
| face_detection_tflite/face_detection_full_range_sparse | OK 12.1 ms | OK 2.38 ms | OK 12.8 ms | OK 5.53 ms | OK 6.72 ms | OK 5.95 ms | MISSING | MISSING | MISSING | MISSING | ACC FAIL 12.7 ms | ERROR 504 | ERROR 504 |
| cat_detection/cat_face_landmarks_full | OK 303.1 ms | OK 225.2 ms | OK 302.6 ms | UNSUPPORTED | UNSUPPORTED | ERROR 3 | MISSING | MISSING | MISSING | MISSING | ERROR 3 | ERROR 3 | ERROR 3 |
| pose_detection/pose_landmark_heavy | OK 109.5 ms | OK 36.8 ms | OK 130.2 ms | ACC FAIL 28.4 ms | OK 39.3 ms | OK 109.6 ms | MISSING | MISSING | MISSING | MISSING | OK 35.1 ms | ACC FAIL 22.9 ms | OK 34.6 ms |
| dog_detection/dog_face_localizer | OK 71.2 ms | OK 21.5 ms | OK 77.4 ms | UNSUPPORTED | UNSUPPORTED | OK 56.4 ms | MISSING | MISSING | MISSING | MISSING | ERROR 3 | ERROR 3 | ERROR 3 |
| face_detection_tflite/face_detection_back | OK 20.1 ms | OK 8.07 ms | OK 19.6 ms | ACC FAIL 6.31 ms | OK 11.1 ms | OK 14.5 ms | MISSING | MISSING | MISSING | MISSING | OK 7.69 ms | OK 6.26 ms | OK 9.13 ms |
| face_detection_tflite/iris_landmark | OK 4.34 ms | OK 1.16 ms | OK 4.57 ms | OK 3.99 ms | OK 3.21 ms | OK 3.18 ms | MISSING | MISSING | MISSING | MISSING | OK 3.41 ms | OK 2.44 ms | OK 3.03 ms |
| face_detection_tflite/face_detection_short_range | OK 3.13 ms | OK 0.81 ms | OK 3.27 ms | ACC FAIL 2.44 ms | OK 3.73 ms | OK 2.56 ms | MISSING | MISSING | MISSING | MISSING | OK 2.36 ms | ACC FAIL 2.45 ms | OK 3.41 ms |

## Device metadata

```json
[
  {
    "schema_version": 1,
    "timestamp_utc": "2026-08-04T00:52:04.612677Z",
    "platform": "android",
    "platform_version": "AP3A.240905.015.A2.A566EXXS2AYD3",
    "abi": "android_arm64",
    "device_model": "SM-A566E",
    "device_manufacturer": "samsung",
    "device_hardware": "s5e8855",
    "device_board": "s5e8855",
    "physical_device": true,
    "physical_device_required": true,
    "device_extra": "SDK 35; ABIs=arm64-v8a",
    "logical_processors": 8,
    "build_mode": "debug",
    "flutter_litert_commit": "800a77b712eba19c432658c32520c1166210e4b2",
    "interpreter_runtime_version": "2.22.0-dev0+selfbuilt",
    "matrix_shard": -1,
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
    "model_count": 29,
    "interpreter_mode_count": 5,
    "compiled_model_mode_count": 4,
    "expected_rows": 261,
    "model_filter": "",
    "mode_filter": "interpreter_cpu_4t,interpreter_xnnpack_4t,interpreter_flex,interpreter_gpu_v2_gl_cl_fp16,interpreter_gpu_v2_gl_cl_fp32,compiled_cpu_fp32,compiled_gpu_cpu_fp32,compiled_gpu_fp16,compiled_gpu_fp32",
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
