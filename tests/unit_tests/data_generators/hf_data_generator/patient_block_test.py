import logging
from datetime import datetime, timedelta

from cehrbert.data_generators.hf_data_generator.patient_block import merge_patient_blocks

# Import or mock your classes
# class Subject: ...
# class PatientBlock: ...


# Create a mock PatientBlock class if you can't import the real one
class MockPatientBlock:
    def __init__(
        self,
        visit_type,
        block_start_time,
        block_end_time,
        has_ed_admission: bool = False,
        has_admission: bool = False,
        discharged_to: str = None,
        events=None,
    ):
        self.visit_type = visit_type
        self.block_start_time = block_start_time
        self.block_end_time = block_end_time
        self.events = events or []
        self.has_ed_admission = has_ed_admission
        self.has_admission = has_admission
        self.has_ed_or_hospital_admission = has_ed_admission | has_admission
        self.discharged_to = discharged_to


# Create a mock Subject class
class MockSubject:
    def __init__(self, subject_id):
        self.subject_id = subject_id


# Create a mock Event class
class MockEvent:
    def __init__(self, time, code):
        self.time = time
        self.code = code


# Configure logging
logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)


def test_merge_patient_blocks():
    # Create test data
    patient = MockSubject(subject_id="test_patient")

    # Create some datetime objects for testing
    now = datetime.now()

    # Scenario 1: One block completely contains another
    block1 = MockPatientBlock(
        visit_type="INPATIENT",
        block_start_time=now,
        block_end_time=now + timedelta(days=5),
        has_admission=True,
        events=[MockEvent(now + timedelta(hours=1), "code1"), MockEvent(now + timedelta(hours=2), "code2")],
    )

    block2 = MockPatientBlock(
        visit_type="OUTPATIENT",
        block_start_time=now + timedelta(days=1),
        block_end_time=now + timedelta(days=2),
        events=[MockEvent(now + timedelta(days=1, hours=3), "code3")],
    )

    # Test Case 1: Merging contained blocks
    patient_blocks = [block1, block2]
    merged_blocks = merge_patient_blocks(patient, patient_blocks)

    # Assertions for test case 1
    assert len(merged_blocks) == 1, "Should merge into one block"
    assert len(merged_blocks[0].events) == 3, "Merged block should have combined events"
    assert merged_blocks[0].visit_type == "INPATIENT", "Should keep the containing block's visit type"

    # Scenario 2: Non-overlapping blocks
    block3 = MockPatientBlock(
        visit_type="EMERGENCY",
        block_start_time=now + timedelta(days=10),
        block_end_time=now + timedelta(days=11),
        has_ed_admission=True,
        events=[
            MockEvent(now + timedelta(days=10, hours=5), "code4"),
        ],
    )

    # Test Case 2: Non-overlapping blocks remain separate
    patient_blocks = [block1, block3]
    merged_blocks = merge_patient_blocks(patient, patient_blocks)

    # Assertions for test case 2
    assert len(merged_blocks) == 2, "Non-overlapping blocks should remain separate"

    # Restored block 1
    block1 = MockPatientBlock(
        visit_type="INPATIENT",
        block_start_time=now,
        block_end_time=now + timedelta(days=5),
        has_admission=True,
        events=[MockEvent(now + timedelta(hours=1), "code1"), MockEvent(now + timedelta(hours=2), "code2")],
    )

    # Test Case 3: Multiple contained blocks
    block4 = MockPatientBlock(
        visit_type="OUTPATIENT",
        block_start_time=now + timedelta(days=3),
        block_end_time=now + timedelta(days=4),
        events=[MockEvent(now + timedelta(days=3, hours=1), "code5")],
    )

    patient_blocks = [block1, block2, block3, block4]
    merged_blocks = merge_patient_blocks(patient, patient_blocks)

    # Assertions for test case 3
    assert len(merged_blocks) == 2, "All contained blocks should merge"
    assert len(merged_blocks[0].events) == 4, "Merged block should have all events from block1, block2, block4"
    assert merged_blocks[1] == block3, "This should be the same as block3"

    # Test Case 4: Empty input
    patient_blocks = []
    merged_blocks = merge_patient_blocks(patient, patient_blocks)

    # Assertions for test case 4
    assert len(merged_blocks) == 0, "Empty input should return empty output"

    print("All tests passed!")

    # Test Case 5: Remove discharge_to and visit_type
    patient_blocks = [
        MockPatientBlock(
            visit_type="INPATIENT",
            block_start_time=now,
            block_end_time=now + timedelta(days=5),
            has_admission=True,
            events=[MockEvent(now + timedelta(hours=1), "code1"), MockEvent(now + timedelta(hours=2), "code2")],
        ),
        MockPatientBlock(
            visit_type="INPATIENT",
            block_start_time=now + timedelta(days=3),
            block_end_time=now + timedelta(days=4),
            discharged_to="discharged_to",
            events=[
                MockEvent(now + timedelta(days=3, hours=1), "INPATIENT"),
                MockEvent(now + timedelta(days=3, hours=2), "code"),
                MockEvent(now + timedelta(days=3, hours=3), "discharged_to"),
            ],
        ),
    ]
    merged_blocks = merge_patient_blocks(patient, patient_blocks)
    # Assertions for test case 4
    assert len(merged_blocks) == 1, "Should only contain one block"
    assert len(merged_blocks[0].events) == 3, "Should contain three events"

    print("All tests passed!")


# Run the test
if __name__ == "__main__":
    test_merge_patient_blocks()
