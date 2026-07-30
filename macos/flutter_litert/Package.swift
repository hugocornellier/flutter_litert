// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "flutter_litert",
    platforms: [
        .macOS("10.15")
    ],
    products: [
        .library(name: "flutter-litert", targets: ["flutter_litert"])
    ],
    dependencies: [
        .package(name: "FlutterFramework", path: "../FlutterFramework")
    ],
    targets: [
        .target(
            name: "flutter_litert",
            dependencies: [
                .product(name: "FlutterFramework", package: "FlutterFramework")
            ],
            resources: [
                .copy("Resources/libtensorflowlite_c-mac.dylib"),
                .copy("Resources/libtflite_custom_ops.dylib"),
                .copy("Resources/libtensorflowlite_gpu-mac.dylib"),
                .copy("Resources/libtensorflowlite_coreml-mac.dylib"),
                .copy("Resources/libLiteRt.dylib"),
                .copy("Resources/libLiteRtMetalAccelerator.dylib"),
                .copy("Resources/libLiteRtCoreMlNpuAccelerator.dylib"),
                .copy("Resources/libtensorflowlite_coreml_npu-mac.dylib"),
                .process("PrivacyInfo.xcprivacy"),
            ]
        )
    ]
)
