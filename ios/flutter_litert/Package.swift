// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "flutter_litert",
    platforms: [
        .iOS("15.0")
    ],
    products: [
        .library(name: "flutter-litert", type: .dynamic, targets: ["flutter_litert"])
    ],
    dependencies: [
        .package(name: "FlutterFramework", path: "../FlutterFramework")
    ],
    targets: [
        // These xcframeworks must be produced with `xcodebuild -create-xcframework`,
        // which writes an Info.plist whose slice identifiers match the on-disk
        // directories. A hand-assembled build previously shipped a mismatch: the
        // Info.plist declared `ios-arm64_x86_64-simulator` while the actual dir and
        // (arm64-only) binary were `ios-arm64-simulator`, so SPM looked for a slice
        // that didn't exist ("There is no XCFramework found at ..."). Each per-slice
        // `.framework` bundle must also contain its own `Info.plist`, or SPM's embed
        // step fails ("framework did not contain an Info.plist"). Both were the
        // root cause of iOS SPM builds breaking; CocoaPods was unaffected because its
        // podspec rewrites the Info.plist at install time.
        //
        // The TensorFlowLite* simulator slices are arm64-only (Bazel
        // `--cpu=ios_sim_arm64`), so iOS-simulator SPM builds work on Apple
        // Silicon only. For x86_64 (Intel) simulator support, also build
        // `--cpu=ios_sim_x86_64`, lipo into a universal simulator slice, and
        // recreate the xcframework. (The LiteRt* targets below already ship
        // universal simulator slices: arm64 plus an empty x86_64 stub, see
        // scripts/wrap_litert_ios_frameworks.sh.)
        .binaryTarget(
            name: "TensorFlowLiteC",
            url: "https://github.com/hugocornellier/flutter_litert/releases/download/tflite-ios-v2.21.0/TensorFlowLiteC-spm.xcframework.zip",
            checksum: "7008f7ed4dd01d2feddf9f45103df71cb157d9383c2964eef283ca6370b54cc2"
        ),
        .binaryTarget(
            name: "TensorFlowLiteCMetal",
            url: "https://github.com/hugocornellier/flutter_litert/releases/download/tflite-ios-v2.21.0/TensorFlowLiteCMetal-spm.xcframework.zip",
            checksum: "22543d2bed3e3f18d90026e21604e1a0e3c298daa5b4695791e806184378ed04"
        ),
        .binaryTarget(
            name: "TensorFlowLiteCCoreML",
            url: "https://github.com/hugocornellier/flutter_litert/releases/download/coreml-ios-v1.2.0/TensorFlowLiteCCoreML-spm.xcframework.zip",
            checksum: "a16b908a36ad17cb520b6a79a8f81aa45b57e8c2abf970746686d3458b41aedd"
        ),
        // LiteRT Next runtime + Metal accelerator (CompiledModel API). These
        // are the same conventional framework-wrapped xcframeworks the
        // CocoaPods channel ships (built by
        // scripts/wrap_litert_ios_frameworks.sh). Earlier releases shipped
        // library-type xcframeworks holding bare dylibs so that LiteRT's
        // GPU-plugin file-name scan (`libLiteRtMetalAccelerator.dylib`) would
        // work, but Xcode embeds those as loose dylibs in the app's
        // Frameworks/ directory, and App Store validation rejects that bundle
        // shape (ITMS-90426, issue #15). The framework rename breaks the
        // file-name scan, so the flutter_litert_gpu_shim target exports
        // LiteRtRegisterGpuAccelerator, which the runtime finds through its
        // RTLD_DEFAULT registration probe instead.
        .binaryTarget(
            name: "LiteRt",
            url: "https://github.com/hugocornellier/flutter_litert/releases/download/litert-ios-v2.0.0/LiteRt-spm.xcframework.zip",
            checksum: "8cb7f1412c78481aa3cd942e0f99c63cc1bc7c97ff3ff8f114d08811e4a384cb"
        ),
        .binaryTarget(
            name: "LiteRtMetalAccelerator",
            url: "https://github.com/hugocornellier/flutter_litert/releases/download/litert-ios-v2.0.0/LiteRtMetalAccelerator-spm.xcframework.zip",
            checksum: "cb6cdab2109fd1a256ff848126bf6bb768f6c564c03e008cce77cf07e2af61f0"
        ),
        .target(
            name: "flutter_litert",
            dependencies: [
                .target(name: "TensorFlowLiteC"),
                .target(name: "TensorFlowLiteCMetal"),
                .target(name: "TensorFlowLiteCCoreML"),
                .target(name: "LiteRt"),
                .target(name: "LiteRtMetalAccelerator"),
                .target(name: "flutter_litert_delegate_symbols"),
                .target(name: "flutter_litert_custom_ops"),
                .target(name: "flutter_litert_gpu_shim"),
                .target(name: "flutter_litert_npu_shim"),
                .product(name: "FlutterFramework", package: "FlutterFramework"),
            ],
            path: "Sources/flutter_litert",
            resources: [
                .process("PrivacyInfo.xcprivacy"),
            ],
            linkerSettings: [
                .linkedFramework("Metal", .when(platforms: [.iOS])),
                .linkedFramework("CoreML", .when(platforms: [.iOS])),
                .linkedFramework("Accelerate", .when(platforms: [.iOS])),
                .linkedLibrary("c++"),
            ]
        ),
        .target(
            name: "flutter_litert_delegate_symbols",
            dependencies: [
                .target(name: "TensorFlowLiteC"),
                .target(name: "TensorFlowLiteCMetal"),
                .target(name: "TensorFlowLiteCCoreML"),
            ],
            path: "Sources/flutter_litert_delegate_symbols",
            publicHeadersPath: "include"
        ),
        .target(
            name: "flutter_litert_custom_ops",
            dependencies: [
                .target(name: "TensorFlowLiteC"),
            ],
            path: "Sources/flutter_litert_custom_ops",
            publicHeadersPath: "include"
        ),
        // Registers the Metal accelerator with the LiteRT Next runtime. The
        // framework-wrapped LiteRt binaries defeat the runtime's bare-dylib
        // file-name scan, so this exports the LiteRtRegisterGpuAccelerator
        // probe target (shared with the CocoaPods channel via
        // ios/Classes/litert_gpu_accelerator_shim.c). No binary-target
        // dependency: the shim talks to the runtime purely through
        // dlopen/dlsym at run time.
        .target(
            name: "flutter_litert_gpu_shim",
            path: "Sources/flutter_litert_gpu_shim",
            publicHeadersPath: "include",
            linkerSettings: [
                .linkedFramework("CoreFoundation", .when(platforms: [.iOS])),
            ]
        ),
        // Registers flutter_litert's patched Core ML delegate as an explicit
        // LiteRT NPU accelerator. The patched CoreML binary is optional at
        // link time so an older cached binary target still builds and reports
        // NPU unsupported instead of failing the whole package link.
        .target(
            name: "flutter_litert_npu_shim",
            path: "Sources/flutter_litert_npu_shim",
            publicHeadersPath: "include"
        )
    ]
)
