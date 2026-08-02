# Android physical-device model matrix

Generated 2026-08-04T00:16:15.527594Z from five Firebase Test Lab Galaxy S23 shards.

- Rows: 261/261 (rectangular: true)
- Status: `{"ok":201,"error":52,"unsupported":8}`
- Accuracy failures: 28
- Native crashes: 0
- Synthesized rows: 0

Each successful cell is `OK p50 ms` or `ACC FAIL p50 ms`. Timing is synchronous invocation with managed I/O for CompiledModel and invoke-only for Interpreter.

| model | interpreter_cpu_4t | interpreter_xnnpack_4t | interpreter_flex | interpreter_gpu_v2_gl_cl_fp16 | interpreter_gpu_v2_gl_cl_fp32 | compiled_cpu_fp32 | compiled_npu_fp32 | compiled_npu_cpu_fp32 | compiled_npu_gpu_cpu_fp32 | compiled_npu_gpu_fp32 | compiled_gpu_cpu_fp32 | compiled_gpu_fp16 | compiled_gpu_fp32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pose_detection/yolov8n_float32 | OK 121.8 ms | OK 67.9 ms | OK 179.6 ms | ACC FAIL 22.7 ms | OK 37.3 ms | OK 159.8 ms | MISSING | MISSING | MISSING | MISSING | OK 39.8 ms | ACC FAIL 26.6 ms | OK 38.5 ms |
| face_detection_tflite/mobilefacenet | OK 40.3 ms | OK 4.05 ms | OK 57.8 ms | OK 6.67 ms | OK 9.11 ms | OK 8.10 ms | MISSING | MISSING | MISSING | MISSING | OK 9.70 ms | ERROR 504 | ERROR 504 |
| hand_detection/hand_detection | OK 64.7 ms | OK 5.99 ms | OK 69.7 ms | ACC FAIL 8.44 ms | OK 10.5 ms | OK 14.2 ms | MISSING | MISSING | MISSING | MISSING | OK 12.8 ms | ACC FAIL 12.6 ms | OK 14.9 ms |
| face_detection_tflite/face_landmark | OK 4.13 ms | OK 1.42 ms | OK 5.99 ms | ACC FAIL 3.59 ms | OK 6.14 ms | OK 2.53 ms | MISSING | MISSING | MISSING | MISSING | OK 11.3 ms | OK 4.90 ms | OK 7.27 ms |
| face_detection_tflite/selfie_segmenter_landscape | OK 28.8 ms | OK 1.45 ms | OK 21.7 ms | ACC FAIL 4.92 ms | OK 8.54 ms | OK 3.76 ms | MISSING | MISSING | MISSING | MISSING | OK 6.01 ms | ACC FAIL 5.31 ms | OK 7.48 ms |
| object_detection/efficientdet_lite2 | OK 174.7 ms | OK 53.3 ms | OK 496.0 ms | ACC FAIL 39.5 ms | OK 47.3 ms | OK 132.9 ms | MISSING | MISSING | MISSING | MISSING | OK 58.8 ms | ACC FAIL 51.4 ms | OK 61.5 ms |
| object_detection/efficientdet_lite0 | OK 276.9 ms | OK 15.1 ms | OK 81.8 ms | ACC FAIL 17.8 ms | OK 22.0 ms | OK 38.4 ms | MISSING | MISSING | MISSING | MISSING | OK 37.9 ms | ACC FAIL 37.6 ms | OK 34.9 ms |
| pose_detection/pose_landmark_lite | OK 63.6 ms | OK 5.11 ms | OK 77.7 ms | ACC FAIL 13.0 ms | OK 14.4 ms | OK 10.5 ms | MISSING | MISSING | MISSING | MISSING | OK 22.4 ms | ACC FAIL 22.8 ms | OK 18.2 ms |
| hand_detection/hand_landmark_full | OK 58.0 ms | OK 5.32 ms | OK 71.1 ms | ACC FAIL 20.2 ms | OK 12.7 ms | OK 11.0 ms | MISSING | MISSING | MISSING | MISSING | OK 11.4 ms | ACC FAIL 11.5 ms | OK 11.0 ms |
| face_detection_tflite/face_detection_front | OK 6.63 ms | OK 1.17 ms | OK 7.93 ms | ACC FAIL 2.52 ms | OK 4.04 ms | OK 2.88 ms | MISSING | MISSING | MISSING | MISSING | OK 4.71 ms | ACC FAIL 2.52 ms | OK 4.33 ms |
| hand_detection/gesture_embedder | OK 0.076 ms | OK 0.037 ms | OK 0.036 ms | OK 1.54 ms | OK 1.53 ms | OK 0.044 ms | MISSING | MISSING | MISSING | MISSING | ERROR 3 | ERROR 3 | ERROR 3 |
| face_detection_tflite/selfie_multiclass | OK 382.2 ms | OK 58.9 ms | OK 184.0 ms | ACC FAIL 22.1 ms | OK 29.1 ms | OK 134.7 ms | MISSING | MISSING | MISSING | MISSING | OK 34.0 ms | ACC FAIL 19.7 ms | OK 32.8 ms |
| animal_detection/superanimal_ssdlite_float16 | OK 93.1 ms | OK 7.68 ms | OK 74.6 ms | OK 11.8 ms | OK 20.4 ms | OK 15.9 ms | MISSING | MISSING | MISSING | MISSING | OK 30.4 ms | ERROR 504 | ERROR 504 |
| pose_detection/pose_landmark_full | OK 156.8 ms | OK 7.23 ms | OK 114.3 ms | ACC FAIL 20.4 ms | OK 14.5 ms | OK 16.3 ms | MISSING | MISSING | MISSING | MISSING | OK 12.7 ms | ACC FAIL 12.0 ms | OK 23.9 ms |
| animal_detection/species_classifier_float16 | OK 23.0 ms | OK 2.74 ms | OK 20.3 ms | ACC FAIL 5.52 ms | OK 7.58 ms | OK 2.85 ms | MISSING | MISSING | MISSING | MISSING | ACC FAIL 12.4 ms | ERROR 504 | ERROR 504 |
| face_detection_tflite/selfie_segmenter | OK 73.5 ms | OK 2.87 ms | OK 48.6 ms | ACC FAIL 5.01 ms | OK 12.9 ms | OK 5.33 ms | MISSING | MISSING | MISSING | MISSING | OK 12.1 ms | ACC FAIL 6.29 ms | OK 13.0 ms |
| dog_detection/dog_face_landmarks_full | OK 416.7 ms | OK 191.4 ms | OK 321.3 ms | UNSUPPORTED | UNSUPPORTED | ERROR 3 | MISSING | MISSING | MISSING | MISSING | ERROR 3 | ERROR 3 | ERROR 3 |
| animal_detection/superanimal_rtmpose_s_float16 | OK 245.0 ms | OK 17.7 ms | OK 78.1 ms | ERROR 1 | ERROR 1 | OK 38.2 ms | MISSING | MISSING | MISSING | MISSING | ERROR 3 | ERROR 504 | ERROR 504 |
| cat_detection/cat_face_localizer | OK 236.5 ms | OK 16.1 ms | OK 178.0 ms | UNSUPPORTED | UNSUPPORTED | OK 38.9 ms | MISSING | MISSING | MISSING | MISSING | ERROR 3 | ERROR 3 | ERROR 3 |
| face_detection_tflite/face_detection_full_range | OK 29.2 ms | OK 2.50 ms | OK 31.4 ms | OK 7.38 ms | OK 8.41 ms | OK 5.17 ms | MISSING | MISSING | MISSING | MISSING | ERROR 3 | ERROR 3 | ERROR 3 |
| face_detection_tflite/face_blendshapes | OK 6.12 ms | OK 0.99 ms | OK 6.48 ms | OK 3.69 ms | OK 3.68 ms | OK 2.77 ms | MISSING | MISSING | MISSING | MISSING | ERROR 3 | ERROR 504 | ERROR 3 |
| hand_detection/canned_gesture_classifier | OK 0.000 ms | OK 0.000 ms | OK 0.001 ms | OK 0.55 ms | OK 0.61 ms | OK 0.004 ms | MISSING | MISSING | MISSING | MISSING | ERROR 3 | ERROR 3 | ERROR 3 |
| face_detection_tflite/face_detection_full_range_sparse | OK 18.2 ms | OK 2.62 ms | OK 17.2 ms | OK 8.57 ms | OK 8.93 ms | OK 4.38 ms | MISSING | MISSING | MISSING | MISSING | ERROR 3 | ERROR 504 | ERROR 504 |
| cat_detection/cat_face_landmarks_full | OK 435.9 ms | OK 175.0 ms | OK 339.1 ms | UNSUPPORTED | UNSUPPORTED | ERROR 3 | MISSING | MISSING | MISSING | MISSING | ERROR 3 | ERROR 3 | ERROR 3 |
| pose_detection/pose_landmark_heavy | OK 378.3 ms | OK 33.2 ms | OK 338.9 ms | ACC FAIL 26.3 ms | OK 23.9 ms | OK 81.7 ms | MISSING | MISSING | MISSING | MISSING | ERROR 3 | ERROR 3 | ERROR 3 |
| dog_detection/dog_face_localizer | OK 223.1 ms | OK 16.3 ms | OK 146.7 ms | UNSUPPORTED | UNSUPPORTED | OK 42.7 ms | MISSING | MISSING | MISSING | MISSING | ERROR 3 | ERROR 3 | ERROR 3 |
| face_detection_tflite/face_detection_back | OK 49.1 ms | OK 4.66 ms | OK 50.8 ms | ACC FAIL 13.5 ms | OK 7.72 ms | OK 11.5 ms | MISSING | MISSING | MISSING | MISSING | ERROR 3 | ERROR 3 | ERROR 3 |
| face_detection_tflite/iris_landmark | OK 12.6 ms | OK 1.18 ms | OK 13.1 ms | OK 4.20 ms | OK 6.77 ms | OK 2.80 ms | MISSING | MISSING | MISSING | MISSING | ERROR 3 | ERROR 3 | ERROR 3 |
| face_detection_tflite/face_detection_short_range | OK 3.11 ms | OK 0.91 ms | OK 5.36 ms | ACC FAIL 3.02 ms | OK 4.35 ms | OK 2.71 ms | MISSING | MISSING | MISSING | MISSING | ERROR 3 | ERROR 3 | ERROR 3 |

## Device metadata

```json
[
  {
    "schema_version": 1,
    "timestamp_utc": "2026-08-04T00:15:12.479118Z",
    "platform": "android",
    "platform_version": "BP1A.250505.005",
    "abi": "android_arm64",
    "device_model": "Pixel 9 Pro",
    "device_manufacturer": "Google",
    "device_hardware": "caiman",
    "device_board": "caiman",
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
