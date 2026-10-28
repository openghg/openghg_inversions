"""Small constructor/instrumentation checks; no sampling or campaign data reads."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from measure import MemoryMonitor, blocked_init, inventory, process_tree
from openghg_inversions.models.cached_sigma import FixedOuCachedSigmaTarget


def main():
    rng = np.random.default_rng(3)
    prepared = SimpleNamespace(n_observation=9, mode_transform=rng.normal(size=(9, 9)))
    for width in (0, 1, 31, 32, 33, 65):
        inputs = dict(
            prepared=prepared,
            observations=rng.normal(size=9),
            fixed_contribution=rng.normal(size=9),
            design=rng.normal(size=(9, width)),
        )
        expected = FixedOuCachedSigmaTarget(**inputs)
        actual = FixedOuCachedSigmaTarget.__new__(FixedOuCachedSigmaTarget)
        blocked_init(actual, **inputs)
        np.testing.assert_allclose(actual._transformed_rhs, expected._transformed_rhs, rtol=1e-13, atol=1e-13)
        assert not np.shares_memory(actual.design, inputs["design"])
    array = np.zeros((3, 4))
    report = inventory({"first": array, "view": array[:1]})
    assert report["retained_unique_backing_bytes"] == array.nbytes
    assert len(report["arrays"]) == 2
    assert process_tree()[0]["pss_bytes"] > 0
    with TemporaryDirectory() as directory:
        path = Path(directory) / "samples.jsonl"
        monitor = MemoryMonitor(path)
        monitor.start()
        monitor.mark("toy_phase")
        summary = monitor.stop()
        samples = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(samples) == sum(phase["samples"] for phase in summary.values())
        assert set(summary) == {"imports_complete", "toy_phase"}
        assert not hasattr(monitor, "samples")  # Telemetry storage stays bounded.
    print("PASS: blocked RHS widths 0/1/31/32/33/65, owned inputs, array deduplication, process PSS")


if __name__ == "__main__":
    main()
