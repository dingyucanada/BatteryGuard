from __future__ import annotations

from batteryguard.demo.generate import generate_demo_dataset


def test_demo_dataset_is_deterministic_and_cell_grouped() -> None:
    first = generate_demo_dataset(seed=42)
    second = generate_demo_dataset(seed=42)
    assert first.cells.equals(second.cells)
    assert first.cycles.equals(second.cycles)
    assert first.splits["cell_id"].is_unique
    assert set(first.splits["split"]) == {
        "train",
        "calibration",
        "test",
        "demo_hidden",
        "external_ood",
    }


def test_demo_only_contains_configured_early_cycles() -> None:
    dataset = generate_demo_dataset(seed=9, early_cycles=30)
    assert dataset.cycles["cycle_index"].max() == 30
    assert len(dataset.splits.query("split == 'demo_hidden'")) == 6
