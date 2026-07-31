pluginManagement {
    val flutterSdkPath =
        run {
            val properties = java.util.Properties()
            file("local.properties").inputStream().use { properties.load(it) }
            val flutterSdkPath = properties.getProperty("flutter.sdk")
            require(flutterSdkPath != null) { "flutter.sdk not set in local.properties" }
            flutterSdkPath
        }

    includeBuild("$flutterSdkPath/packages/flutter_tools/gradle")

    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

plugins {
    id("dev.flutter.flutter-plugin-loader") version "1.0.0"
    id("com.android.application") version "8.11.1" apply false
    id("com.android.dynamic-feature") version "8.11.1" apply false
    id("com.android.library") version "8.11.1" apply false
    id("org.jetbrains.kotlin.android") version "2.2.20" apply false
}

include(":app")

providers.gradleProperty("flutterLitert.qualcommNpuFeatureRoot").orNull?.let { value ->
    val root = file(value)
    require(root.isAbsolute) {
        "flutterLitert.qualcommNpuFeatureRoot must be an absolute path"
    }
    require(root.isDirectory) {
        "flutterLitert.qualcommNpuFeatureRoot does not exist: $root"
    }

    val runtimeStrings = root.resolve("runtime_strings")
    require(runtimeStrings.isDirectory) {
        "Missing LiteRT runtime_strings module: $runtimeStrings"
    }
    include(":litert_npu_runtime_strings")
    project(":litert_npu_runtime_strings").projectDir = runtimeStrings

    listOf("73", "75", "79").forEach { htpVersion ->
        val module = ":qualcomm_runtime_v$htpVersion"
        val moduleDirectory = root.resolve("qualcomm_runtime_v$htpVersion")
        require(moduleDirectory.isDirectory) {
            "Missing Qualcomm HTP v$htpVersion module: $moduleDirectory"
        }
        include(module)
        project(module).projectDir = moduleDirectory
    }
}
