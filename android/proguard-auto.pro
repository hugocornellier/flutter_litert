# Consumer R8 rules for flutterLitert.runtime=auto, on top of
# proguard-play-services.pro.
#
# play-services-litert raises the litert-api that litert-gpu 1.4.2 depends on
# to 2.x, which no longer has org.tensorflow.lite.Delegate, so R8 stops at the
# Java GpuDelegate that references it. The plugin never loads that class: Dart
# binds the bundled runtime's native libraries directly.
-dontwarn org.tensorflow.lite.Delegate
