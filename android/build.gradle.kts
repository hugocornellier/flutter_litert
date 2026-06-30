import java.io.File
import java.net.URI
import javax.inject.Inject
import org.gradle.api.DefaultTask
import org.gradle.api.GradleException
import org.gradle.api.file.ArchiveOperations
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.file.FileSystemOperations
import org.gradle.api.provider.Property
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.OutputDirectory
import org.gradle.api.tasks.TaskAction
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.dsl.KotlinAndroidProjectExtension

group = "com.hugocornellier.flutter_litert"
version = "1.0"

buildscript {
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        classpath("com.android.tools.build:gradle:8.13.2")
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:2.3.20")
    }
}

rootProject.allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

plugins {
    id("com.android.library")
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
    compileSdk = 36
    namespace = "com.hugocornellier.flutter_litert"

    externalNativeBuild {
        cmake {
            path = file("../src/CMakeLists.txt")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        minSdk = 21
    }
}

// When a Kotlin compile task exists (KGP on AGP < 9, or built-in Kotlin on
// AGP 8.11+/9), it defaults its JVM target to the JDK version (e.g. 21/22)
// and clashes with the Java 17 target. Pin it to 17. Guarded because on
// AGP 9 without KGP the kotlin extension is not registered.
extensions.findByType<KotlinAndroidProjectExtension>()?.apply {
    compilerOptions {
        jvmTarget = JvmTarget.JVM_17
    }
}

dependencies {
    val litert = "1.4.2"
    add("implementation", "com.google.ai.edge.litert:litert:$litert")
    add("implementation", "com.google.ai.edge.litert:litert-gpu:$litert")
}

// LiteRT Next runtime (CompiledModel API) libLiteRt.so, sourced directly from
// Google's official Maven AAR (com.google.ai.edge.litert:litert:<ver>, the 2.x
// line) instead of a hand-packaged release zip. Downloaded at build time (not
// shipped in the pub package) and extracted as a jniLib, additively alongside
// the classic litert:1.4.x artifact above — the Interpreter path is unchanged.
//
// The 2.x AAR ships libLiteRt.so for arm64-v8a, armeabi-v7a AND x86_64 (the
// GPU accelerator, libLiteRtClGlAccelerator.so, ships for arm64-v8a/x86_64
// only). armeabi-v7a support comes for free here — the old hand-packaged zip
// was arm64-v8a/x86_64 only.
//
// Why fetch+extract rather than a normal Gradle `implementation` dependency:
// the classic Interpreter still needs litert:1.4.x (libtensorflowlite_jni.so),
// and the 2.x AAR dropped that .so in favour of libLiteRt.so. Declaring both
// versions of the same module would resolve to a single version and break one
// of the two runtimes. Extracting only libLiteRt.so keeps them side by side.
val litertNextVersion = "2.1.5"

// libLiteRt.so is contributed to the build as a GENERATED jniLibs source through
// the AGP Variant API (androidComponents.onVariants -> jniLibs
// .addGeneratedSourceDirectory). This is the AGP-9 replacement for the old
// `android.sourceSets["main"].jniLibs.srcDir(<Provider>)` wiring: AGP 9 defaults
// `android.sourceset.disallowProvider` to true and rejects passing a Provider to
// the legacy SourceSet API. Modelling it as a generated source also lets AGP own
// the task dependency (no manual preBuild hook) and makes a litertNextVersion bump
// re-download, because the version is a tracked task input.
abstract class DownloadLitertJniTask : DefaultTask() {
    @get:Input
    abstract val litertVersion: Property<String>

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @get:Inject
    abstract val archiveOperations: ArchiveOperations

    @get:Inject
    abstract val fileSystemOperations: FileSystemOperations

    @TaskAction
    fun download() {
        val version = litertVersion.get()
        val outDir = outputDir.get().asFile
        val marker = File(outDir, "arm64-v8a/libLiteRt.so")
        val aar = File(temporaryDir, "litert-$version.aar")
        val url = "https://dl.google.com/dl/android/maven2/com/google/ai/edge/" +
            "litert/litert/$version/litert-$version.aar"
        logger.lifecycle("[flutter_litert] Downloading LiteRT Next runtime (libLiteRt.so) from Maven $version...")
        fileSystemOperations.delete {
            delete(outDir)
            delete(aar)
        }
        outDir.mkdirs()
        aar.parentFile.mkdirs()
        URI(url).toURL().openStream().use { input ->
            aar.outputStream().use { output -> input.copyTo(output) }
        }
        fileSystemOperations.copy {
            from(archiveOperations.zipTree(aar)) {
                // CPU runtime only for now. libLiteRtClGlAccelerator.so is in
                // the same AAR (arm64-v8a/x86_64) but stays unbundled: when it
                // registers in an environment without working OpenCL (every
                // emulator), even GPU|CPU-fallback compilation fails with a
                // runtime error instead of falling back. Bundle it once the GPU
                // path is validated on real hardware.
                include("jni/**/libLiteRt.so")
                // Strip the AAR's leading "jni/" so files land at the jniLibs
                // layout <abi>/libLiteRt.so.
                eachFile { path = path.substringAfter("jni/") }
                includeEmptyDirs = false
            }
            into(outDir)
        }
        if (!marker.exists()) {
            throw GradleException("[flutter_litert] LiteRT Next Maven AAR did not yield $marker")
        }
    }
}

val downloadLitertJni = tasks.register<DownloadLitertJniTask>("downloadLitertJni") {
    litertVersion.set(litertNextVersion)
}

androidComponents {
    onVariants { variant ->
        val jniLibs = checkNotNull(variant.sources.jniLibs) {
            "[flutter_litert] AGP did not expose jniLibs sources for variant ${variant.name}"
        }
        jniLibs.addGeneratedSourceDirectory(downloadLitertJni) { it.outputDir }
    }
}
