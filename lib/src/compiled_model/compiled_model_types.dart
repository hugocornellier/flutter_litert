/*
 * Copyright 2026 flutter_litert authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/// Option types shared by every [CompiledModel] platform implementation
/// (native FFI, web LiteRT.js, unsupported stub). Each implementation file
/// re-exports these so the conditional export in `compiled_model.dart`
/// resolves the same names everywhere.
library;

/// Hardware accelerator requested for LiteRT Next compilation.
///
/// On the web, [cpu] maps to the LiteRT.js WASM backend and [gpu] to WebGPU;
/// [npu] is native-only.
enum Accelerator { cpu, gpu, npu }

/// GPU precision mode for LiteRT Next compilation.
enum Precision { fp16, fp32 }

/// Tensor buffer allocation mode for CompiledModel I/O.
///
/// [managed] uses LiteRT-managed buffers and copies host data through the
/// documented lock/write/unlock and lock/read/unlock path. [hostMemory] wraps
/// package-owned, 64-byte-aligned host memory with
/// `LiteRtCreateTensorBufferFromHostMemory`.
///
/// Host-memory buffers are opt-in because performance is model- and
/// accelerator-dependent: they can reduce lock/copy overhead for some strict
/// GPU models, but are slower for others. [hostMemory] is native-only; the web
/// implementation supports [managed] buffers exclusively.
enum TensorBufferMode { managed, hostMemory }
