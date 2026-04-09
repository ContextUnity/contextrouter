from contextrouter.cortex.runtime_context import (
    append_provenance,
    get_accumulated_provenance,
    init_provenance_accumulator,
    reset_provenance_accumulator,
)


def test_provenance_accumulator_lifecycle():
    assert get_accumulated_provenance() == []

    # Init accumulator
    token = init_provenance_accumulator()
    assert get_accumulated_provenance() == []

    # Append basic items
    append_provenance("tool:search")
    assert get_accumulated_provenance() == ["tool:search"]

    # Append complex items
    append_provenance(("auth", "check"))
    assert get_accumulated_provenance() == ["tool:search", ("auth", "check")]

    # Reset accumulator
    reset_provenance_accumulator(token)
    assert get_accumulated_provenance() == []

    # Append after reset (graceful no-op)
    append_provenance("should_not_append")
    assert get_accumulated_provenance() == []
