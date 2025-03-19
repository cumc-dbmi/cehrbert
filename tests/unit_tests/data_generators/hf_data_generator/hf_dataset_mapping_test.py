import datetime
import unittest

from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import VisitObject, has_events_and_get_events


def test_visit_object_sorting():
    """Test that VisitObject instances sort chronologically by start datetime."""
    # Create visits with different start dates
    visit1 = VisitObject(
        visit_start_datetime=datetime.datetime(2023, 3, 15, 9, 0),
        visit_end_datetime=datetime.datetime(2023, 3, 15, 10, 30),
        visit_type="OUTPATIENT",
        discharge_facility="Main Clinic",
        events=[],
    )

    visit2 = VisitObject(
        visit_start_datetime=datetime.datetime(2023, 1, 10, 8, 0),
        visit_end_datetime=datetime.datetime(2023, 1, 12, 14, 0),
        visit_type="INPATIENT",
        discharge_facility="General Hospital",
        events=[],
    )

    visit3 = VisitObject(
        visit_start_datetime=datetime.datetime(2023, 5, 22, 15, 30),
        visit_end_datetime=datetime.datetime(2023, 5, 22, 16, 0),
        visit_type="EMERGENCY",
        discharge_facility="Emergency Center",
        events=[],
    )

    visit4 = VisitObject(
        visit_start_datetime=datetime.datetime(2022, 11, 5, 10, 15),
        visit_end_datetime=datetime.datetime(2022, 11, 5, 11, 45),
        visit_type="OUTPATIENT",
        discharge_facility="Satellite Clinic",
        events=[],
    )

    # Store visits in a non-chronological order
    unsorted_visits = [visit1, visit2, visit3, visit4]

    # Expected chronological order (earliest to latest)
    expected_chronological = [visit4, visit2, visit1, visit3]

    # Test sorted() function
    sorted_visits = sorted(unsorted_visits)

    # Check if the order matches the expected chronological order
    assert len(sorted_visits) == len(expected_chronological)
    for i, (actual, expected) in enumerate(zip(sorted_visits, expected_chronological)):
        assert (
            actual.visit_start_datetime == expected.visit_start_datetime
        ), f"Visit at position {i} has incorrect datetime: {actual.visit_start_datetime} != {expected.visit_start_datetime}"


def test_has_events_and_get_events():
    """Consolidated test function for has_events_and_get_events."""
    # Test case 1: None input
    has_events, events = has_events_and_get_events(None)
    assert has_events is False, "Should return False for None input"
    assert events is None, "Should return None for None input"
    print("✓ Test passed: None input")

    # Test case 2: Empty list
    has_events, events = has_events_and_get_events([])
    assert has_events is False, "Should return False for empty list"
    assert events is None, "Should return None for empty list"
    print("✓ Test passed: Empty list")

    # Test case 3: Empty generator
    def empty_generator():
        yield from []

    has_events, events = has_events_and_get_events(empty_generator())
    assert has_events is False, "Should return False for empty generator"
    assert events is None, "Should return None for empty generator"
    print("✓ Test passed: Empty generator")

    # Test case 4: Single item list
    original = ["event1"]
    has_events, events = has_events_and_get_events(original)
    assert has_events is True, "Should return True for single item list"
    assert events is not None, "Should return an iterable for single item list"
    events_list = list(events)
    assert len(events_list) == len(original), "Returned iterable should have same length as original"
    assert events_list == original, "Returned iterable should contain same elements as original"
    print("✓ Test passed: Single item list")

    # Test case 5: Multiple items list
    original = ["event1", "event2", "event3"]
    has_events, events = has_events_and_get_events(original)
    assert has_events is True, "Should return True for multiple items list"
    assert events is not None, "Should return an iterable for multiple items list"
    events_list = list(events)
    assert len(events_list) == len(original), "Returned iterable should have same length as original"
    assert events_list == original, "Returned iterable should contain same elements as original"
    print("✓ Test passed: Multiple items list")

    # Test case 6: Generator with items
    def event_generator():
        yield "event1"
        yield "event2"
        yield "event3"
        yield "event4"

    original = list(event_generator())
    has_events, events = has_events_and_get_events(event_generator())
    assert has_events is True, "Should return True for generator with items"
    assert events is not None, "Should return an iterable for generator with items"
    events_list = list(events)
    assert len(events_list) == len(original), "Returned iterable should have same length as original"
    assert events_list == original, "Returned iterable should contain same elements as original"
    print("✓ Test passed: Generator with items")

    # Test case 7: Non-iterable input
    has_events, events = has_events_and_get_events(42)
    assert has_events is False, "Should return False for non-iterable input"
    assert events is None, "Should return None for non-iterable input"
    print("✓ Test passed: Non-iterable input")

    # Test case 8: Custom generator similar to event_generator
    def custom_event_generator():
        times = [1, 2, 3]
        codes = ["A", "B", "C"]
        for time, code in zip(times, codes):
            yield f"{code}-{time}"

    original = list(custom_event_generator())
    has_events, events = has_events_and_get_events(custom_event_generator())
    assert has_events is True, "Should return True for custom generator"
    assert events is not None, "Should return an iterable for custom generator"
    events_list = list(events)
    assert len(events_list) == len(original), "Returned iterable should have same length as original"
    assert events_list == original, "Returned iterable should contain same elements as original"
    print("✓ Test passed: Custom generator")

    # Test case 9: Integer elements
    original = [1, 2, 3, 4, 5]
    has_events, events = has_events_and_get_events(original)
    assert has_events is True, "Should return True for list of integers"
    assert events is not None, "Should return an iterable for list of integers"
    events_list = list(events)
    assert len(events_list) == len(original), "Returned iterable should have same length as original"
    assert events_list == original, "Returned iterable should contain same elements as original"
    print("✓ Test passed: Integer elements")

    print("\nAll tests passed!")
    return True


if __name__ == "__main__":
    unittest.main()
