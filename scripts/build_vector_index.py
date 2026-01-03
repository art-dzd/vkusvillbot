from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vkusvillbot.config import Settings
from vkusvillbot.db import Database
from vkusvillbot.embeddings_client import OpenRouterEmbeddingsClient
from vkusvillbot.vector_index import FaissVectorIndex, VectorIndexError


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Сборка FAISS индекса по товарам ВкусВилл")
    parser.add_argument(
        "--db",
        default=None,
        help="Путь к SQLite БД (по умолчанию из CONFIG/.env)",
    )
    parser.add_argument(
        "--index",
        default=None,
        help="Путь к FAISS индексу (по умолчанию из CONFIG/.env)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenRouter модель эмбеддингов (по умолчанию из CONFIG/.env)",
    )
    parser.add_argument("--batch-size", type=int, default=48, help="Размер батча для эмбеддингов")
    parser.add_argument("--sleep", type=float, default=0.2, help="Пауза между батчами (сек)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Пересчитать эмбеддинги даже если content_hash не изменился",
    )
    return parser.parse_args()


def _embedding_bytes(vec: list[float]) -> bytes:
    return np.asarray(vec, dtype="float32").tobytes()


def _decode_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="float32")


async def build() -> None:
    args = _parse_args()
    settings = Settings.load()
    db_path = args.db or settings.db.path
    index_path = Path(args.index or settings.vector.index_path)
    embedding_model = args.model or settings.vector.embedding_model

    if not settings.llm.api_key:
        raise RuntimeError("OPENROUTER_API_KEY не задан")

    db = Database(db_path)
    try:
        if db.has_products():
            db.ensure_product_columns()
            db.ensure_fts()
        else:
            raise RuntimeError("В БД нет таблицы products")

        products = db.list_products_for_embedding()
        if not products:
            raise RuntimeError("Товары не найдены")

        texts: dict[int, str] = {}
        hashes: dict[int, str] = {}
        for item in products:
            product_id = int(item["id"])
            text = db.embedding_text(item)
            texts[product_id] = text
            hashes[product_id] = db.embedding_hash(text)

        existing = db.get_existing_embedding_hashes(list(texts.keys()), embedding_model)

        to_update: list[int] = []
        for product_id, content_hash in hashes.items():
            if args.force:
                to_update.append(product_id)
                continue
            if existing.get(product_id) != content_hash:
                to_update.append(product_id)

        print(
            f"Товаров: {len(texts)} | "
            f"к обновлению эмбеддингов: {len(to_update)} | "
            f"модель: {embedding_model}"
        )

        client = OpenRouterEmbeddingsClient(
            api_key=settings.llm.api_key,
            model=embedding_model,
            referer=settings.llm.http_referer,
            title=settings.llm.title,
            proxy_url=settings.llm.proxy_url,
        )

        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        batch_size = max(1, int(args.batch_size))
        sleep_s = max(0.0, float(args.sleep))

        for start in range(0, len(to_update), batch_size):
            batch_ids = to_update[start : start + batch_size]
            batch_texts = [texts[i] for i in batch_ids]
            embeddings = await client.embed(batch_texts)
            if len(embeddings) != len(batch_ids):
                raise RuntimeError(
                    f"OpenRouter вернул {len(embeddings)} эмбеддингов на {len(batch_ids)} текстов"
                )
            rows = []
            for product_id, emb in zip(batch_ids, embeddings, strict=True):
                rows.append(
                    (product_id, embedding_model, hashes[product_id], _embedding_bytes(emb), ts)
                )
            db.upsert_embeddings(rows)
            done = min(len(to_update), start + batch_size)
            print(f"Эмбеддинги: {done}/{len(to_update)}")
            if sleep_s:
                await asyncio.sleep(sleep_s)

        stored = db.load_embeddings(embedding_model)
        if not stored:
            raise RuntimeError("В БД нет эмбеддингов для построения индекса")

        ids: list[int] = []
        vecs: list[np.ndarray] = []
        dim: int | None = None
        for product_id, blob, _ in stored:
            vec = _decode_embedding(blob)
            if dim is None:
                dim = int(vec.shape[0])
            if vec.shape[0] != dim:
                continue
            ids.append(product_id)
            vecs.append(vec)

        if dim is None:
            raise RuntimeError("Не удалось определить размерность эмбеддингов")

        vectors = np.vstack(vecs).astype("float32", copy=False)
        index = FaissVectorIndex(index_path)
        try:
            index.build_and_save(ids, vectors)
        except VectorIndexError as exc:
            raise RuntimeError(str(exc)) from exc

        meta = {
            "built_at": ts,
            "embedding_model": embedding_model,
            "dim": dim,
            "count": len(ids),
            "db_path": str(Path(db_path).resolve()),
            "index_path": str(index_path),
        }
        meta_path = index_path.with_suffix(index_path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"Индекс сохранён: {index_path} | товаров в индексе: {len(ids)} | dim={dim}")
        print(f"Метаданные: {meta_path}")
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(build())
