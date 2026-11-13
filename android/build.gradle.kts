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

// Which LiteRT runtime the plugin ships, set by the consuming app in
// android/gradle.properties:
//
// flutterLitert.runtime=bundled       (default) LiteRT is packaged in the APK.
// flutterLitert.runtime=playServices  LiteRT comes from Google Play services
//                                     (play-services-litert), so no LiteRT
//                                     native library is packaged. The app must
//                                     call initializeAndroidRuntime() first.
// flutterLitert.runtime=auto          Both are packaged, the bundled runtime as
//                                     native libraries only.
//                                     initializeAndroidRuntime() uses Play
//                                     services when it can provide both runtime
//                                     halves, and the bundled runtime otherwise.
val litertRuntime =
    providers.gradleProperty("flutterLitert.runtime").orNull?.let { value ->
        when (value) {
            "bundled", "playServices", "auto" -> value
            else -> throw GradleException(
                "[flutter_litert] flutterLitert.runtime must be bundled, " +
                    "playServices or auto, not '$value'"
            )
        }
    } ?: "bundled"
// Whether the Play services client is linked (playServices and auto).
val usePlayServicesRuntime = litertRuntime != "bundled"
// Whether LiteRT itself is packaged in the APK (bundled and auto).
val packageBundledRuntime = litertRuntime != "playServices"
val playServicesLitertVersion = "17.0.0"

android {
    compileSdk = 36
    namespace = "com.hugocornellier.flutter_litert"

    externalNativeBuild {
        cmake {
            path = file("../src/CMakeLists.txt")
        }
    }

    if (usePlayServicesRuntime) {
        // libflutter_litert_runtime.so links the C client that
        // play-services-litert ships as a prefab package.
        buildFeatures {
            prefab = true
        }
    }

    sourceSets {
        getByName("main") {
            java.srcDir(
                file(if (usePlayServicesRuntime) "src/playServices/java" else "src/bundled/java")
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        // libLiteRt.so and play-services-litert both require API 24.
        minSdk = 24
        if (usePlayServicesRuntime) {
            // Keeps release builds of apps that use the runtime from crashing
            // at launch; see the file.
            consumerProguardFiles("proguard-play-services.pro")
        }
        externalNativeBuild {
            cmake {
                arguments += "-DFLUTTER_LITERT_RUNTIME=$litertRuntime"
            }
        }
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
    if (usePlayServicesRuntime) {
        add(
            "implementation",
            "com.google.android.gms:play-services-litert:$playServicesLitertVersion"
        )
    }
}

// LiteRT native libraries (Interpreter and CompiledModel), sourced directly from
// Google's official Maven AARs (com.google.ai.edge.litert:litert and
// litert-gpu, the 2.x line) instead of a hand-packaged release zip. Downloaded
// at build time (not shipped in the pub package) and extracted as jniLibs.
//
// The litert AAR ships libLiteRt.so for arm64-v8a, armeabi-v7a, and x86_64.
// The litert-gpu AAR ships the GPU accelerator, libLiteRtClGlAccelerator.so,
// for the same three ABIs. We package it for arm64-v8a and x86_64 only, by
// choice: the armeabi-v7a accelerator is untested here, and armeabi-v7a stays
// fully supported on the CPU.
//
// Why fetch+extract rather than a normal Gradle `implementation` dependency:
// Dart binds these libraries directly and never needs the AAR's Java, and
// flutterLitert.runtime=auto cannot link that Java alongside
// play-services-litert.
//
// Only the .so files are extracted, so the AARs' LICENSE and
// THIRD_PARTY_NOTICE.txt (identical in litert and litert-gpu 2.3.2) ship
// byte-for-byte in android/licenses/litert-<version>/ instead. When this
// version changes, replace that folder with the new AAR's two files.
val litertNextVersion = "2.3.2"

// The GPU accelerator is bundled by default. Apps that do not use it can set
// flutterLitert.bundleGpuAccelerator=false in android/gradle.properties.
val shouldBundleGpuAccelerator =
    providers.gradleProperty("flutterLitert.bundleGpuAccelerator").orNull?.let { value ->
        value.toBooleanStrictOrNull()
            ?: throw GradleException(
                "[flutter_litert] flutterLitert.bundleGpuAccelerator must be true or false"
            )
    } ?: true

// Android NPU runtime libraries are vendor- and SoC-specific. Google publishes
// the LiteRT side of them as separate Maven AARs
// (com.google.ai.edge.litert:litert-npu-runtime-qualcomm and the other
// litert-npu-runtime-* artifacts), and Qualcomm ships the QAIRT libraries in
// its own SDK. Normal builds stay unchanged. A consuming app (or the
// physical-device test workflow) may point this property at one prepared
// Qualcomm JIT runtime directory containing the compiler and dispatch
// libraries from litert-npu-runtime-qualcomm (same version as litert) and the
// matching QAIRT libraries:
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

    // Without timeouts a stalled connection hangs the build until whatever
    // runs it gives up, and one dropped connection fails it. The Linux and
    // Windows CMake builds try their downloads three times as well.
    private fun downloadWithRetry(url: String, target: File) {
        val attempts = 3
        for (attempt in 1..attempts) {
            try {
                val connection = URI(url).toURL().openConnection().apply {
                    connectTimeout = 30_000
                    readTimeout = 60_000
                }
                connection.getInputStream().use { input ->
                    target.outputStream().use { output -> input.copyTo(output) }
                }
                return
            } catch (e: java.io.IOException) {
                target.delete()
                if (attempt == attempts) {
                    throw GradleException(
                        "[flutter_litert] Failed to download $url after $attempts attempts",
                        e
                    )
                }
                logger.warn("[flutter_litert] Download of $url failed ($e); retrying")
                Thread.sleep(5_000L * attempt)
            }
        }
    }

    @TaskAction
    fun download() {
        val version = litertVersion.get()
        val shouldBundleGpu = bundleGpuAccelerator.get()
        val qualcommNpuDir = qualcommNpuRuntimeDir.orNull?.asFile
        val outDir = outputDir.get().asFile
        // Since 2.3.0 the GPU accelerator ships in litert-gpu, not litert.
        val artifacts = buildList {
            add("litert")
            if (shouldBundleGpu) add("litert-gpu")
        }
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
        outDir.mkdirs()
        temporaryDir.mkdirs()
        artifacts.forEach { artifact ->
            val aar = File(temporaryDir, "$artifact-$version.aar")
            val url = "https://dl.google.com/dl/android/maven2/com/google/ai/edge/" +
                "litert/$artifact/$version/$artifact-$version.aar"
            fileSystemOperations.delete {
                delete(aar)
            }
            downloadWithRetry(url, aar)
            fileSystemOperations.copy {
                from(archiveOperations.zipTree(aar)) {
                    // Android emulators do not provide working OpenCL. If this
                    // accelerator registers but cannot initialize, a combined GPU
                    // and CPU compilation can fail instead of falling back inside
                    // LiteRT. The Dart fallback factories catch that error and
                    // retry CPU-only.
                    include("jni/**/libLiteRt.so")
                    // litert-gpu also ships an armeabi-v7a accelerator; it is
                    // left out on purpose (see the comment on litertNextVersion).
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
                "[flutter_litert] LiteRT Next Maven AARs yielded an invalid " +
                    "native library set: ${problems.joinToString()}"
            )
        }
    }
}

// Android flex was removed in 4.0.0. flutter_litert_flex 1.x has no
// constraint on flutter_litert, so pub can still resolve it next to 4.0.0,
// and its Android side needs the LiteRT 1.x libtensorflowlite_jni.so that
// 4.0.0 no longer ships. Flutter includes every plugin that has Android code
// as the Gradle project :<plugin name>, so this finds 1.x in any app, whatever
// the runtime. A flutter_litert_flex without Android code has no such project
// and passes.
if (rootProject.findProject(":flutter_litert_flex") != null) {
    throw GradleException(
        "[flutter_litert] Android flex was removed in flutter_litert 4.0.0 " +
            "because LiteRT 2.x has no flex delegate (Select TF ops), but " +
            "this app includes flutter_litert_flex for Android. Reconvert " +
            "the model to LiteRT built-in ops and remove flutter_litert_flex, " +
            "or stay on flutter_litert 3.10.x with flutter_litert_flex 1.3.0."
    )
}

if (!packageBundledRuntime) {
    // These options configure the bundled runtime's native libraries, which
    // the Play services runtime does not package. Reject them rather than
    // silently ignoring an explicit setting.
    val bundledOnlyProperties = listOf(
        "flutterLitert.bundleGpuAccelerator",
        "flutterLitert.qualcommNpuRuntimeDir"
    ).filter { providers.gradleProperty(it).isPresent }
    if (bundledOnlyProperties.isNotEmpty()) {
        throw GradleException(
            "[flutter_litert] ${bundledOnlyProperties.joinToString()} only " +
                "apply when the bundled runtime is packaged " +
                "(flutterLitert.runtime=bundled or auto). Remove them to use " +
                "the Google Play services runtime."
        )
    }
} else {
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
}
