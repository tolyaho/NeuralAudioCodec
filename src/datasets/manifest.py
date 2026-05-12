"""Build LibriSpeech manifest files."""

import json
from pathlib import Path

import soundfile as sf


def find_audio_files(split_dir: Path) -> list[Path]:
    return sorted(split_dir.rglob("*.flac"))


def read_transcripts(split_dir: Path) -> dict[str, str]:
    transcripts = {}

    for path in sorted(split_dir.rglob("*.trans.txt")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                utt_id, text = line.split(" ", 1)
                transcripts[utt_id] = text

    return transcripts


def parse_librispeech_id(audio_path: Path) -> tuple[str, str, str]:
    utt_id = audio_path.stem
    speaker_id, chapter_id, *_ = utt_id.split("-")
    return utt_id, speaker_id, chapter_id


def get_audio_info(audio_path: Path) -> tuple[int, int, float]:
    info = sf.info(str(audio_path))

    sample_rate = int(info.samplerate)
    num_frames = int(info.frames)
    duration = num_frames / sample_rate

    return sample_rate, num_frames, duration


def make_manifest_item(
    audio_path: Path,
    root: Path,
    split: str,
    transcripts: dict[str, str],
) -> dict:
    utt_id, speaker_id, chapter_id = parse_librispeech_id(audio_path)
    sample_rate, num_frames, duration = get_audio_info(audio_path)

    return {
        "utt_id": utt_id,
        "speaker_id": speaker_id,
        "chapter_id": chapter_id,
        "sample_rate": sample_rate,
        "num_frames": num_frames,
        "duration": duration,
        "text": transcripts.get(utt_id, ""),
        "audio_path": str(audio_path.relative_to(root)),
        "split": split,
    }


def write_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def summarize_manifest(items: list[dict]) -> dict:
    if not items:
        return {
            "num_utterances": 0,
            "num_speakers": 0,
            "total_hours": 0.0,
            "min_duration": 0.0,
            "mean_duration": 0.0,
            "max_duration": 0.0,
            "sample_rates": [],
            "missing_text": 0,
        }

    durations = [item["duration"] for item in items]
    speakers = {item["speaker_id"] for item in items}
    sample_rates = {item["sample_rate"] for item in items}

    return {
        "num_utterances": len(items),
        "num_speakers": len(speakers),
        "total_hours": sum(durations) / 3600,
        "min_duration": min(durations),
        "mean_duration": sum(durations) / len(durations),
        "max_duration": max(durations),
        "sample_rates": sorted(sample_rates),
        "missing_text": sum(item["text"] == "" for item in items),
    }


def build_split_manifest(root: Path, split: str, out_dir: Path) -> dict:
    split_dir = root / "data" / "raw" / "LibriSpeech" / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    transcripts = read_transcripts(split_dir)
    audio_files = find_audio_files(split_dir)
    items = [make_manifest_item(audio_path, root, split, transcripts) for audio_path in audio_files]

    out_path = root / out_dir / f"{split}.jsonl"
    write_jsonl(items, out_path)

    summary = summarize_manifest(items)
    summary["manifest_path"] = str(out_path.relative_to(root))

    return summary
