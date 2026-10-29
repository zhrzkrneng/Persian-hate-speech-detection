"""Reader for the Hateful Memes Challenge release.

The data is not in this repository and must not be added to it. See
`scripts/download_hmc.sh`.

Expected layout under `data/hmc/`:

    train.jsonl  dev_seen.jsonl  dev_unseen.jsonl  test_seen.jsonl
    img/<id>.png

Each JSONL line carries `id`, `img`, `text`, and — on the labelled splits —
`label` (0 non-hateful, 1 hateful).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemeRecord:
    uid: str
    image_path: Path
    text: str
    label: int | None


class HatefulMemesSplit:
    def __init__(self, root: str | Path, split: str) -> None:
        self.root = Path(root)
        self.split = split
        self.records = self._read()

    def _read(self) -> list[MemeRecord]:
        path = self.root / f"{self.split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. The Hateful Memes data is distributed by "
                "Meta under a click-through licence and is not shipped with "
                "this repository. Run scripts/download_hmc.sh for instructions."
            )

        records: list[MemeRecord] = []
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
                records.append(
                    MemeRecord(
                        uid=str(row["id"]),
                        image_path=self.root / row["img"],
                        text=row["text"],
                        label=row.get("label"),
                    )
                )
        if not records:
            raise ValueError(f"{path} is empty")
        return records

    def missing_images(self) -> list[Path]:
        return [r.image_path for r in self.records if not r.image_path.exists()]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> MemeRecord:
        return self.records[index]
