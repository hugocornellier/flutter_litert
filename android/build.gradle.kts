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
import org.gradle.api.tasks.InputDirectory
import org.gradle.api.tasks.Optional
import org.gradle.api.tasks.OutputDirectory
import org.gradle.api.tasks.PathSensitive
import org.gradle.api.tasks.PathSensitivity
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

// LiteRT Next native libraries (CompiledModel API), sourced directly from
// Google's official Maven AAR (com.google.ai.edge.litert:litert:<ver>, the 2.x
// line) instead of a hand-packaged release zip. Downloaded at build time (not
// shipped in the pub package) and extracted as jniLibs alongside the classic
// litert:1.4.x artifact above. The Interpreter path is unchanged.
//
// The 2.x AAR ships libLiteRt.so for arm64-v8a, armeabi-v7a, and x86_64. Its
// GPU accelerator, libLiteRtClGlAccelerator.so, ships for arm64-v8a and x86_64
// only, so armeabi-v7a remains fully supported on the CPU.
//
// Why fetch+extract rather than a normal Gradle `implementation` dependency:
// the classic Interpreter still needs litert:1.4.x (libtensorflowlite_jni.so),
// and the 2.x AAR dropped that .so in favour of libLiteRt.so. Declaring both
// versions of the same module would resolve to a single version and break one
// of the two runtimes. Selectively extracting the 2.x native libraries keeps
// them side by side.
val litertNextVersion = "2.1.6"

// The GPU accelerator is bundled by default. Apps that do not use it can set
// flutterLitert.bundleGpuAccelerator=false in android/gradle.properties.
val shouldBundleGpuAccelerator =
    providers.gradleProperty("flutterLitert.bundleGpuAccelerator").orNull?.let { value ->
        value.toBooleanStrictOrNull()
            ?: throw GradleException(
                "[flutter_litert] flutterLitert.bundleGpuAccelerator must be true or false"
            )
    } ?: true

// Android NPU runtime libraries are vendor- and SoC-specific, and Google
// distributes them separately from the LiteRT Maven AAR. Normal builds stay
// unchanged. A consuming app (or the physical-device test workflow) may point
// this property at one prepared Qualcomm JIT runtime directory containing the
// matching LiteRT compiler/dispatch libraries and QAIRT libraries:
//
// flutterLitert.qualcommNpuRuntimeDir=/absolute/path/to/arm64-v8a
//
// The app must also use legacy JNI packaging so LiteRT can scan the extracted
// directory. Production apps should prefer Google's conditional Play Feature
// Delivery modules; this direct bundle is primarily for local/Firebase APK
// validation and deliberately supports exactly one Qualcomm HTP generation.
val qualcommNpuRuntimeDirectory =
    providers.gradleProperty("flutterLitert.qualcommNpuRuntimeDir").orNull?.let { value ->
        val directory = File(value)
        if (!directory.isAbsolute) {
            throw GradleException(
                "[flutter_litert] flutterLitert.qualcommNpuRuntimeDir must be " +
                    "an absolute path"
            )
        }
        directory
    }

// The LiteRT Next libraries are contributed as a GENERATED jniLibs source through
// the AGP Variant API (androidComponents.onVariants -> jniLibs
// .addGeneratedSourceDirectory). This is the AGP-9 replacement for the old
// `android.sourceSets["main"].jniLibs.srcDir(<Provider>)` wiring: AGP 9 defaults
// `android.sourceset.disallowProvider` to true and rejects passing a Provider to
// the legacy SourceSet API. Modelling it as a generated source also lets AGP own
// the task dependency (no manual preBuild hook). The LiteRT version and GPU bundle
// flag are tracked task inputs, so changing either one regenerates the shared output.
abstract class DownloadLitertJniTask : DefaultTask() {
    @get:Input
    abstract val litertVersion: Property<String>

    @get:Input
    abstract val bundleGpuAccelerator: Property<Boolean>

    @get:Optional
    @get:InputDirectory
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val qualcommNpuRuntimeDir: DirectoryProperty

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @get:Inject
    abstract val archiveOperations: ArchiveOperations

    @get:Inject
    abstract val fileSystemOperations: FileSystemOperations

    @TaskAction
    fun download() {
        val version = litertVersion.get()
        val shouldBundleGpu = bundleGpuAccelerator.get()
        val qualcommNpuDir = qualcommNpuRuntimeDir.orNull?.asFile
        val outDir = outputDir.get().asFile
        val aar = File(temporaryDir, "litert-$version.aar")
        val url = "https://dl.google.com/dl/android/maven2/com/google/ai/edge/" +
            "litert/litert/$version/litert-$version.aar"
        val libraryNames = buildList {
            add("libLiteRt.so")
            if (shouldBundleGpu) add("libLiteRtClGlAccelerator.so")
            if (qualcommNpuDir != null) add("Qualcomm NPU JIT runtime")
        }.joinToString()
        logger.lifecycle(
            "[flutter_litert] Downloading LiteRT Next native libraries " +
                "($libraryNames) from Maven $version..."
        )
        if (outDir.exists() && !outDir.deleteRecursively()) {
            throw GradleException(
                "[flutter_litert] Failed to clean generated JNI directory: $outDir"
            )
        }
        fileSystemOperations.delete {
            delete(aar)
        }
        outDir.mkdirs()
        aar.parentFile.mkdirs()
        URI(url).toURL().openStream().use { input ->
            aar.outputStream().use { output -> input.copyTo(output) }
        }
        fileSystemOperations.copy {
            from(archiveOperations.zipTree(aar)) {
                // Android emulators do not provide working OpenCL. If this
                // accelerator registers but cannot initialize, a combined GPU
                // and CPU compilation can fail instead of falling back inside
                // LiteRT. The Dart fallback factories catch that error and
                // retry CPU-only.
                include("jni/**/libLiteRt.so")
                if (shouldBundleGpu) {
                    include(
                        "jni/arm64-v8a/libLiteRtClGlAccelerator.so",
                        "jni/x86_64/libLiteRtClGlAccelerator.so"
                    )
                }
                // Strip the AAR's leading "jni/" so files land at the jniLibs
                // layout <abi>/libLiteRt*.so.
                eachFile { path = path.substringAfter("jni/") }
                includeEmptyDirs = false
            }
            into(outDir)
        }

        val qualcommNpuLibraries = if (qualcommNpuDir != null) {
            val names = qualcommNpuDir.listFiles()
                ?.filter { it.isFile }
                ?.map { it.name }
                ?.toSet()
                ?: throw GradleException(
                    "[flutter_litert] Cannot list Qualcomm NPU runtime directory: " +
                        qualcommNpuDir
                )
            val htpVersionPattern = Regex("""libQnnHtpV(69|73|75|79|81)Skel[.]so""")
            val htpVersions = names.mapNotNull { name ->
                htpVersionPattern.matchEntire(name)?.groupValues?.get(1)
            }
            if (htpVersions.size != 1) {
                throw GradleException(
                    "[flutter_litert] Qualcomm NPU runtime must contain exactly " +
                        "one supported HTP Skel library (v69, v73, v75, v79, or " +
                        "v81); found ${htpVersions.size} in $qualcommNpuDir"
                )
            }
            val htpVersion = htpVersions.single()
            val required = listOf(
                "libLiteRtCompilerPlugin_Qualcomm.so",
                "libLiteRtDispatch_Qualcomm.so",
                "libQnnHtp.so",
                "libQnnSystem.so",
                "libQnnHtpV${htpVersion}Skel.so",
                "libQnnHtpV${htpVersion}Stub.so",
                "libQnnHtpPrepare.so",
                "libQnnIr.so",
                "libQnnSaver.so"
            )
            val missing = required.filterNot(names::contains)
            if (missing.isNotEmpty()) {
                throw GradleException(
                    "[flutter_litert] Qualcomm NPU JIT runtime is incomplete: " +
                        missing.joinToString { "missing $it" }
                )
            }
            val conflictingLiteRtLibraries = names.filter { name ->
                (name.startsWith("libLiteRtCompilerPlugin_") ||
                    name.startsWith("libLiteRtDispatch_")) &&
                    name !in required
            }
            if (conflictingLiteRtLibraries.isNotEmpty()) {
                throw GradleException(
                    "[flutter_litert] Qualcomm NPU runtime contains conflicting " +
                        "LiteRT vendor libraries: " +
                        conflictingLiteRtLibraries.sorted().joinToString()
                )
            }

            fileSystemOperations.copy {
                from(qualcommNpuDir) {
                    include(required)
                }
                into(File(outDir, "arm64-v8a"))
            }
            required
        } else {
            emptyList()
        }

        val expectedLibraries = mutableListOf(
            "arm64-v8a/libLiteRt.so",
            "armeabi-v7a/libLiteRt.so",
            "x86_64/libLiteRt.so"
        )
        if (shouldBundleGpu) {
            expectedLibraries += listOf(
                "arm64-v8a/libLiteRtClGlAccelerator.so",
                "x86_64/libLiteRtClGlAccelerator.so"
            )
        }
        expectedLibraries += qualcommNpuLibraries.map { "arm64-v8a/$it" }
        val unexpectedLibraries = buildList {
            if (shouldBundleGpu) {
                add("armeabi-v7a/libLiteRtClGlAccelerator.so")
            } else {
                addAll(
                    listOf(
                        "arm64-v8a/libLiteRtClGlAccelerator.so",
                        "armeabi-v7a/libLiteRtClGlAccelerator.so",
                        "x86_64/libLiteRtClGlAccelerator.so"
                    )
                )
            }
            if (qualcommNpuDir == null) {
                File(outDir, "arm64-v8a").listFiles()
                    ?.filter { file ->
                        file.name.startsWith("libLiteRtCompilerPlugin_") ||
                            file.name.startsWith("libLiteRtDispatch_") ||
                            file.name.startsWith("libQnn")
                    }
                    ?.forEach { file -> add("arm64-v8a/${file.name}") }
            }
        }
        val missing = expectedLibraries.filterNot { File(outDir, it).isFile }
        val unexpected = unexpectedLibraries.filter { File(outDir, it).exists() }
        if (missing.isNotEmpty() || unexpected.isNotEmpty()) {
            val problems = missing.map { "missing $it" } +
                unexpected.map { "unexpected $it" }
            throw GradleException(
                "[flutter_litert] LiteRT Next Maven AAR yielded an invalid " +
                    "native library set: ${problems.joinToString()}"
            )
        }
    }
}

val downloadLitertJni = tasks.register<DownloadLitertJniTask>("downloadLitertJni") {
    litertVersion.set(litertNextVersion)
    bundleGpuAccelerator.set(shouldBundleGpuAccelerator)
    if (qualcommNpuRuntimeDirectory != null) {
        qualcommNpuRuntimeDir.fileValue(qualcommNpuRuntimeDirectory)
    }
}

androidComponents {
    onVariants { variant ->
        val jniLibs = checkNotNull(variant.sources.jniLibs) {
            "[flutter_litert] AGP did not expose jniLibs sources for variant ${variant.name}"
        }
        jniLibs.addGeneratedSourceDirectory(downloadLitertJni) { it.outputDir }
    }
}
