// Install-time feature module for Qualcomm HTP v75 (SM8650). The workflow
// fills src/main/jni/arm64-v8a with the LiteRT compiler and dispatch
// libraries from litert-npu-runtime-qualcomm and the matching QAIRT libraries.
plugins { id("com.android.dynamic-feature") }

android {
    namespace = "com.google.ai.edge.litert.qualcomm_runtime.v75"
    compileSdk = 35

    defaultConfig { minSdk = 31 }

    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("src/main/jni")
        }
    }
}

dependencies { implementation(project(":app")) }
