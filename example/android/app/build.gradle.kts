import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.dsl.KotlinAndroidProjectExtension

plugins {
    id("com.android.application")
    id("dev.flutter.flutter-gradle-plugin")
}

val testLabAbi = providers.gradleProperty("flutterLitert.testLabAbi").orNull
val qualcommNpuRuntimeDir =
    providers.gradleProperty("flutterLitert.qualcommNpuRuntimeDir").orNull
val qualcommNpuFeatureRoot =
    providers.gradleProperty("flutterLitert.qualcommNpuFeatureRoot").orNull

require(qualcommNpuRuntimeDir == null || qualcommNpuFeatureRoot == null) {
    "Use either flutterLitert.qualcommNpuRuntimeDir for a fused APK or " +
        "flutterLitert.qualcommNpuFeatureRoot for device-targeted delivery, not both"
}

val agpVersion = com.android.Version.ANDROID_GRADLE_PLUGIN_VERSION
    .substringBefore('.')
    .toInt()
val builtInKotlinProperty = providers.gradleProperty("android.builtInKotlin").orNull
val isBuiltInKotlinEnabled = agpVersion >= 9 && (builtInKotlinProperty == null || builtInKotlinProperty.toBoolean())
val shouldApplyKotlinAndroidPlugin = agpVersion < 9 || !isBuiltInKotlinEnabled

if (shouldApplyKotlinAndroidPlugin) {
    apply(plugin = "org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.flutter_litert_example"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        applicationId = "com.example.flutter_litert_example"
        // The official Qualcomm dispatch modules require Android 12. Normal
        // example builds retain Flutter's lower minimum SDK.
        minSdk = if (qualcommNpuFeatureRoot != null) 31 else flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // Physical Test Lab devices are arm64. Keep normal example/emulator
        // builds multi-ABI, but avoid uploading a several-hundred-MB fat debug
        // APK for the dedicated cloud test.
        if (testLabAbi != null) {
            require(testLabAbi == "arm64-v8a") {
                "flutterLitert.testLabAbi only supports arm64-v8a"
            }
            ndk {
                abiFilters.clear()
                abiFilters.add(testLabAbi)
            }
        }
    }

    buildTypes {
        release {
            // Use debug signing so local release runs work without a keystore.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    if (qualcommNpuRuntimeDir != null || qualcommNpuFeatureRoot != null) {
        // LiteRT discovers the NPU compiler and dispatch plugins by scanning
        // ApplicationInfo.nativeLibraryDir. Qualcomm builds therefore must
        // extract JNI libraries instead of loading them directly from the APK.
        packaging {
            jniLibs {
                useLegacyPackaging = true
            }
        }
    }

    if (qualcommNpuFeatureRoot != null) {
        bundle {
            deviceTargetingConfig = file("device_targeting_configuration.xml")
            deviceGroup {
                enableSplit = true
                defaultGroup = "other"
            }
        }
        dynamicFeatures.addAll(
            setOf(
                ":qualcomm_runtime_v73",
                ":qualcomm_runtime_v75",
                ":qualcomm_runtime_v79",
            )
        )
    }
}

flutter {
    source = "../.."
}

dependencies {
    if (qualcommNpuFeatureRoot != null) {
        implementation(project(":litert_npu_runtime_strings"))
    }
    testImplementation("junit:junit:4.13.2")
    // Match the versions resolved by Flutter's integration_test plugin on the
    // app runtime classpath; AGP enforces consistent debug/androidTest graphs.
    androidTestImplementation("androidx.test:runner:1.3.0")
    androidTestImplementation("androidx.test:rules:1.2.0")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.3.0")
}

extensions.findByType<KotlinAndroidProjectExtension>()?.apply {
    compilerOptions {
        jvmTarget = JvmTarget.JVM_17
    }
}
