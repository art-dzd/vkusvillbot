from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # noqa: BLE001
    faiss = None  # type: ignore


class VectorIndexError(RuntimeError):
    pass


@dataclass(frozen=True)
class VectorHit:
    id: int
    score: float


class FaissVectorIndex:
    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self._index: object | None = None
        self._mtime: float | None = None

    def available(self) -> bool:
        return self.index_path.exists()

    def _require_faiss(self) -> None:
        if faiss is None:
            raise VectorIndexError(
                "faiss не установлен. Установите зависимость faiss-cpu и пересоберите контейнер."
            )

    def load(self) -> None:
        self._require_faiss()
        if not self.index_path.exists():
            raise VectorIndexError(
                f"FAISS индекс не найден: {self.index_path}. "
                "Соберите индекс скриптом scripts/build_vector_index.py."
            )
        mtime = self.index_path.stat().st_mtime
        if self._index is not None and self._mtime == mtime:
            return
        self._index = faiss.read_index(str(self.index_path))
        self._mtime = mtime

    def search(self, vector: np.ndarray, k: int) -> list[VectorHit]:
        self._require_faiss()
        self.load()
        if k < 1:
            return []
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        vector = vector.astype("float32", copy=False)
        faiss.normalize_L2(vector)
        distances, ids = self._index.search(vector, k)  # type: ignore[attr-defined]
        result: list[VectorHit] = []
        for raw_id, raw_score in zip(ids[0].tolist(), distances[0].tolist(), strict=False):
            if raw_id == -1:
                continue
            result.append(VectorHit(id=int(raw_id), score=float(raw_score)))
        return result

    def build_and_save(self, ids: Iterable[int], vectors: np.ndarray) -> None:
        self._require_faiss()
        ids_list = [int(v) for v in ids]
        if not ids_list:
            raise VectorIndexError("Нет данных для построения индекса")
        vectors = vectors.astype("float32", copy=False)
        if vectors.ndim != 2:
            raise VectorIndexError("vectors должен быть 2D массивом")
        if vectors.shape[0] != len(ids_list):
            raise VectorIndexError("Количество ids и vectors не совпадает")

        dim = int(vectors.shape[1])
        base = faiss.IndexFlatIP(dim)
        index = faiss.IndexIDMap2(base)
        faiss.normalize_L2(vectors)
        index.add_with_ids(vectors, np.asarray(ids_list, dtype="int64"))

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        self._index = index
        self._mtime = self.index_path.stat().st_mtime

