// Titles for the Qualcomm NPU feature modules next to this one. LiteRT 2.2.0
// shipped this scaffolding in its GitHub NPU zip; from 2.3.0 the libraries
// come from Google Maven, which has none, so it lives here. The
// android-npu-testlab workflow copies this folder into the directory it
// passes as flutterLitert.qualcommNpuFeatureRoot.
plugins { id("com.android.library") }

android {
    namespace = "com.google.ai.edge.litert.runtime_strings"
    compileSdk = 35

    defaultConfig { minSdk = 31 }
}
