# Consumer R8 rules for flutterLitert.runtime=playServices.
#
# play-services-litert depends on com.google.android.play:asset-delivery, which
# brings in androidx.work 2.9 and, through it, androidx.room 2.5. Room creates
# WorkDatabase_Impl by reflection, but the keep rule Room 2.5 ships keeps the
# class without its constructor. R8 full mode, the default since AGP 8, then
# removes the constructor, and WorkManager's startup initializer crashes the
# app at launch ("Failed to create an instance of class
# androidx.work.impl.WorkDatabase"). Room 2.6 and later ship this same rule.
-keep class * extends androidx.room.RoomDatabase { <init>(); }
