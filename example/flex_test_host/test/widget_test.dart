import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:flutter_litert_flex_test_host/main.dart';

void main() {
  testWidgets('renders the Flex integration-test host', (tester) async {
    await tester.pumpWidget(const FlexTestHostApp());

    expect(find.byType(FlexTestHostApp), findsOneWidget);
    expect(find.byType(SizedBox), findsOneWidget);
    expect(tester.takeException(), isNull);
  });
}
