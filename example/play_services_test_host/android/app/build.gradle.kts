plugins {
    id("com.android.application")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

// Firebase Test Lab runs on arm64 phones; building one ABI keeps the APK small.
val testLabAbi = providers.gradleProperty("flutterLitert.testLabAbi").orNull

android {
    namespace = "com.hugocornellier.flutter_litert_play_services_test_host"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        applicationId = "com.hugocornellier.flutter_litert_play_services_test_host"
        // play-services-litert requires API 24.
        minSdk = maxOf(24, flutter.minSdkVersion)
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

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
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

flutter {
    source = "../.."
}

dependencies {
    // Firebase Test Lab instrumentation (see MainActivityTest).
    androidTestImplementation("androidx.test:runner:1.3.0")
    androidTestImplementation("androidx.test:rules:1.2.0")
}
