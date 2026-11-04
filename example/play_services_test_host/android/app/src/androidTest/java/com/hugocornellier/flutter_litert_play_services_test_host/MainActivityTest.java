package com.hugocornellier.flutter_litert_play_services_test_host;

import androidx.test.rule.ActivityTestRule;
import dev.flutter.plugins.integration_test.FlutterTestRunner;
import org.junit.Rule;
import org.junit.runner.RunWith;

/** Runs the Dart integration-test target as an Android instrumentation test. */
@RunWith(FlutterTestRunner.class)
public final class MainActivityTest {
  @Rule
  public ActivityTestRule<MainActivity> rule =
      new ActivityTestRule<>(MainActivity.class, true, false);
}
