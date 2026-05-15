import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.manifest import build_split_manifest


def print_summary(split: str, summary: dict) -> None:
    print(f"\n[{split}]")
    print(f"manifest: {summary['manifest_path']}")
    print(f"utterances: {summary['num_utterances']}")
    print(f"speakers: {summary['num_speakers']}")
    print(f"hours: {summary['total_hours']:.2f}")
    print(
        "duration: "
        f"{summary['min_duration']:.2f}s / "
        f"{summary['mean_duration']:.2f}s / "
        f"{summary['max_duration']:.2f}s "
        "(min/mean/max)"
    )
    print(f"sample rates: {summary['sample_rates']}")
    print(f"missing text: {summary['missing_text']}")


def main() -> None:
    splits = ["train-clean-100", "test-clean"]
    out_dir = Path("data/manifests")
    summaries = {}

    for split in splits:
        summary = build_split_manifest(
            root=ROOT,
            split=split,
            out_dir=out_dir,
        )

        summaries[split] = summary
        print_summary(split, summary)

    summary_path = ROOT / out_dir / "summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print(f"\nBuilt the manifests and saved summaries to {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
