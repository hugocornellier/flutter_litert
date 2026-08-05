# Android physical-device model matrix

Generated 2026-08-05T10:58:04.319685Z from five Firebase Test Lab Galaxy S23 shards.

- Rows: 377/377 (rectangular: true)
- Status: `{"ok":321,"error":48,"unsupported":8}`
- Accuracy failures: 80
- Native crashes: 0
- Synthesized rows: 0

Each successful cell is `OK p50 ms` or `ACC FAIL p50 ms`. Timing is synchronous invocation with managed I/O for CompiledModel and invoke-only for Interpreter.

| model | interpreter_cpu_4t | interpreter_xnnpack_4t | interpreter_flex | interpreter_gpu_v2_gl_cl_fp16 | interpreter_gpu_v2_gl_cl_fp32 | compiled_cpu_fp32 | compiled_npu_fp32 | compiled_npu_cpu_fp32 | compiled_npu_gpu_cpu_fp32 | compiled_npu_gpu_fp32 | compiled_gpu_cpu_fp32 | compiled_gpu_fp16 | compiled_gpu_fp32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pose_detection/yolov8n_float32 | OK 112.5 ms | OK 69.5 ms | OK 114.4 ms | ACC FAIL 17.4 ms | OK 29.7 ms | OK 159.8 ms | ACC FAIL 18.5 ms | ACC FAIL 18.7 ms | ACC FAIL 17.9 ms | ACC FAIL 17.7 ms | OK 18.8 ms | ACC FAIL 11.9 ms | OK 18.6 ms |
| face_detection_tflite/mobilefacenet | OK 15.3 ms | OK 3.70 ms | OK 15.3 ms | OK 2.41 ms | OK 3.28 ms | OK 8.89 ms | OK 6.01 ms | OK 5.91 ms | OK 5.72 ms | OK 6.05 ms | OK 2.27 ms | ERROR 504 | ERROR 504 |
| hand_detection/hand_detection | OK 17.5 ms | OK 6.23 ms | OK 22.9 ms | ACC FAIL 3.33 ms | OK 4.76 ms | OK 14.7 ms | OK 4.58 ms | OK 4.61 ms | OK 4.66 ms | OK 4.39 ms | OK 3.31 ms | ACC FAIL 2.87 ms | OK 3.20 ms |
| face_detection_tflite/face_landmark | OK 2.67 ms | OK 0.88 ms | OK 3.60 ms | ACC FAIL 1.06 ms | OK 1.27 ms | OK 1.74 ms | OK 3.18 ms | OK 3.02 ms | OK 3.19 ms | OK 3.17 ms | OK 1.36 ms | OK 0.98 ms | OK 1.62 ms |
| face_detection_tflite/selfie_segmenter_landscape | OK 10.8 ms | OK 0.86 ms | OK 10.8 ms | ACC FAIL 1.58 ms | OK 1.90 ms | OK 1.72 ms | ERROR 504 | OK 1.72 ms | OK 1.59 ms | OK 1.20 ms | OK 1.49 ms | ACC FAIL 1.15 ms | OK 1.59 ms |
| object_detection/efficientdet_lite2 | OK 110.7 ms | OK 56.1 ms | OK 155.0 ms | ACC FAIL 19.7 ms | OK 30.8 ms | OK 125.7 ms | ACC FAIL 42.0 ms | ACC FAIL 37.4 ms | ACC FAIL 40.4 ms | ACC FAIL 39.5 ms | OK 32.2 ms | ACC FAIL 24.7 ms | OK 33.0 ms |
| object_detection/efficientdet_lite0 | OK 39.4 ms | OK 16.1 ms | OK 55.9 ms | ACC FAIL 9.43 ms | OK 13.1 ms | OK 40.0 ms | ACC FAIL 14.1 ms | ACC FAIL 13.9 ms | ACC FAIL 14.9 ms | ACC FAIL 15.5 ms | OK 13.6 ms | ACC FAIL 10.2 ms | OK 13.8 ms |
| pose_detection/pose_landmark_lite | OK 15.5 ms | OK 4.08 ms | OK 20.1 ms | ACC FAIL 4.07 ms | OK 5.61 ms | OK 10.3 ms | ACC FAIL 8.20 ms | ACC FAIL 8.48 ms | ACC FAIL 7.47 ms | ACC FAIL 8.21 ms | OK 4.48 ms | ACC FAIL 3.64 ms | OK 5.51 ms |
| hand_detection/hand_landmark_full | OK 13.2 ms | OK 4.87 ms | OK 15.9 ms | ACC FAIL 4.01 ms | OK 5.37 ms | OK 11.7 ms | ACC FAIL 4.14 ms | ACC FAIL 4.18 ms | ACC FAIL 4.06 ms | ACC FAIL 4.14 ms | OK 3.85 ms | ACC FAIL 2.66 ms | OK 3.45 ms |
| face_detection_tflite/face_detection_front | OK 2.45 ms | OK 1.14 ms | OK 2.65 ms | ACC FAIL 0.84 ms | OK 1.34 ms | OK 1.49 ms | OK 2.99 ms | OK 2.99 ms | OK 2.85 ms | OK 3.01 ms | OK 1.09 ms | ACC FAIL 0.98 ms | OK 1.15 ms |
| hand_detection/gesture_embedder | OK 0.029 ms | OK 0.024 ms | OK 0.029 ms | ACC FAIL 0.56 ms | OK 0.62 ms | OK 0.019 ms | ERROR 3 | ERROR 3 | ERROR 3 | ERROR 3 | ERROR 3 | ERROR 3 | ERROR 3 |
| face_detection_tflite/selfie_multiclass | OK 97.3 ms | OK 63.3 ms | OK 111.9 ms | ACC FAIL 12.7 ms | OK 21.9 ms | OK 134.1 ms | ACC FAIL 14.9 ms | ACC FAIL 14.0 ms | ACC FAIL 13.9 ms | ACC FAIL 13.3 ms | OK 16.7 ms | ACC FAIL 10.0 ms | OK 16.7 ms |
| animal_detection/superanimal_ssdlite_float16 | OK 23.8 ms | OK 7.90 ms | OK 27.8 ms | OK 5.31 ms | OK 7.10 ms | OK 16.6 ms | OK 7.09 ms | OK 7.15 ms | OK 7.56 ms | OK 7.44 ms | OK 16.4 ms | ERROR 504 | ERROR 504 |
| pose_detection/pose_landmark_full | OK 24.9 ms | OK 7.29 ms | OK 29.7 ms | ACC FAIL 5.79 ms | OK 8.16 ms | OK 17.1 ms | ACC FAIL 9.07 ms | ACC FAIL 9.27 ms | ACC FAIL 9.19 ms | ACC FAIL 9.35 ms | OK 7.20 ms | ACC FAIL 4.51 ms | OK 7.01 ms |
| animal_detection/species_classifier_float16 | OK 5.08 ms | OK 1.63 ms | OK 4.21 ms | ACC FAIL 2.49 ms | OK 2.74 ms | OK 2.84 ms | ACC FAIL 4.17 ms | ACC FAIL 4.43 ms | ACC FAIL 4.38 ms | ACC FAIL 4.51 ms | ACC FAIL 5.20 ms | ERROR 504 | ERROR 504 |
| face_detection_tflite/selfie_segmenter | OK 18.9 ms | OK 1.54 ms | OK 19.0 ms | ACC FAIL 1.70 ms | OK 2.80 ms | OK 3.36 ms | ERROR 504 | OK 3.35 ms | OK 2.25 ms | OK 2.28 ms | OK 2.20 ms | ACC FAIL 1.92 ms | OK 2.34 ms |
| dog_detection/dog_face_landmarks_full | OK 192.9 ms | OK 144.9 ms | OK 174.1 ms | UNSUPPORTED | UNSUPPORTED | ERROR 3 | ERROR 504 | OK 167.3 ms | OK 168.2 ms | ERROR 504 | ERROR 3 | ERROR 3 | ERROR 3 |
| animal_detection/superanimal_rtmpose_s_float16 | OK 34.3 ms | OK 13.9 ms | OK 34.3 ms | ACC FAIL 9.56 ms | ACC FAIL 12.6 ms | OK 35.0 ms | ACC FAIL 4.51 ms | ACC FAIL 4.66 ms | ACC FAIL 4.77 ms | ACC FAIL 4.61 ms | ACC FAIL 34.8 ms | ERROR 504 | ERROR 504 |
| cat_detection/cat_face_localizer | OK 51.2 ms | OK 16.3 ms | OK 54.1 ms | UNSUPPORTED | UNSUPPORTED | OK 39.4 ms | ERROR 504 | ACC FAIL 158.0 ms | ACC FAIL 158.8 ms | ERROR 504 | ERROR 3 | ERROR 3 | ERROR 3 |
| face_detection_tflite/face_detection_full_range | OK 7.64 ms | OK 1.98 ms | OK 6.59 ms | OK 2.31 ms | OK 2.87 ms | OK 4.76 ms | OK 3.79 ms | OK 3.82 ms | OK 3.79 ms | OK 3.82 ms | OK 2.43 ms | OK 1.56 ms | OK 2.79 ms |
| face_detection_tflite/face_blendshapes | OK 1.73 ms | OK 0.82 ms | OK 1.74 ms | OK 1.16 ms | OK 1.40 ms | OK 1.02 ms | OK 4.09 ms | OK 4.07 ms | OK 4.07 ms | OK 4.07 ms | ERROR 3 | ERROR 504 | ERROR 3 |
| hand_detection/canned_gesture_classifier | OK 0.000 ms | OK 0.000 ms | OK 0.000 ms | OK 0.34 ms | OK 0.37 ms | OK 0.001 ms | OK 2.58 ms | OK 2.59 ms | OK 2.58 ms | OK 2.61 ms | OK 0.42 ms | OK 0.34 ms | OK 0.33 ms |
| face_detection_tflite/face_detection_full_range_sparse | OK 7.38 ms | OK 1.96 ms | OK 8.24 ms | OK 2.17 ms | OK 3.15 ms | OK 4.08 ms | ERROR 3 | ERROR 3 | ERROR 3 | ERROR 3 | ACC FAIL 5.62 ms | ERROR 504 | ERROR 504 |
| cat_detection/cat_face_landmarks_full | OK 188.6 ms | OK 142.6 ms | OK 175.3 ms | UNSUPPORTED | UNSUPPORTED | ERROR 3 | ERROR 504 | OK 170.5 ms | OK 169.2 ms | ERROR 504 | ERROR 3 | ERROR 3 | ERROR 3 |
| pose_detection/pose_landmark_heavy | OK 69.9 ms | OK 30.9 ms | OK 94.2 ms | ACC FAIL 13.6 ms | OK 19.6 ms | OK 69.7 ms | ACC FAIL 12.3 ms | ACC FAIL 12.5 ms | ACC FAIL 11.5 ms | ACC FAIL 12.6 ms | OK 14.4 ms | ACC FAIL 9.49 ms | OK 15.2 ms |
| dog_detection/dog_face_localizer | OK 48.1 ms | OK 16.5 ms | OK 50.8 ms | UNSUPPORTED | UNSUPPORTED | OK 40.3 ms | ERROR 504 | ACC FAIL 159.6 ms | ACC FAIL 157.4 ms | ERROR 504 | ERROR 3 | ERROR 3 | ERROR 3 |
| face_detection_tflite/face_detection_back | OK 13.9 ms | OK 4.03 ms | OK 16.6 ms | ACC FAIL 2.13 ms | OK 3.51 ms | OK 9.90 ms | OK 4.54 ms | OK 4.62 ms | OK 4.58 ms | OK 4.49 ms | OK 2.57 ms | ACC FAIL 1.49 ms | OK 3.19 ms |
| face_detection_tflite/iris_landmark | OK 2.94 ms | OK 0.90 ms | OK 3.03 ms | OK 2.44 ms | OK 2.94 ms | OK 2.80 ms | OK 2.98 ms | OK 2.89 ms | OK 2.98 ms | OK 2.96 ms | OK 2.03 ms | OK 1.49 ms | OK 1.85 ms |
| face_detection_tflite/face_detection_short_range | OK 2.38 ms | OK 0.72 ms | OK 1.84 ms | ACC FAIL 1.05 ms | OK 1.25 ms | OK 1.63 ms | OK 2.92 ms | OK 2.92 ms | OK 2.98 ms | OK 2.95 ms | OK 1.11 ms | ACC FAIL 0.96 ms | OK 1.12 ms |

## Device metadata

```json
[
  {
    "schema_version": 1,
    "timestamp_utc": "2026-08-05T10:56:31.428231Z",
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
    "flutter_litert_commit": "6bd62453cd508004536c90d34779b2e758327957",
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
    "compiled_model_mode_count": 8,
    "expected_rows": 377,
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
