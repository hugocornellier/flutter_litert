// Install-time feature module for Qualcomm HTP v79 (SM8750). The workflow
// fills src/main/jni/arm64-v8a with the LiteRT compiler and dispatch
// libraries from litert-npu-runtime-qualcomm and the matching QAIRT libraries.
plugins { id("com.android.dynamic-feature") }

android {
    namespace = "com.google.ai.edge.litert.qualcomm_runtime.v79"
    compileSdk = 35

    defaultConfig { minSdk = 31 }

    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("src/main/jni")
        }
    }
}

dependencies { implementation(project(":app")) }
