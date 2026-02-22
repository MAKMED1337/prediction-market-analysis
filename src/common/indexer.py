"""
Base class for data indexers that fetch and store market data.

Usage:
    from src.common.indexer import Indexer

    class MyIndexer(Indexer):
        def run(self) -> None:
            # Fetch and store data
            pass

    indexer = MyIndexer("my_indexer", "Fetches data from source")
    indexer.run()
"""

from __future__ import annotations

import importlib
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")
U = TypeVar("U")


class Indexer(ABC):
    """Base class for data indexers.

    Subclasses implement `run()` to fetch and store data.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self) -> None:
        """Execute the indexer to fetch and store data."""
        pass

    @classmethod
    def load(cls, indexer_dir: Path | str = "src/indexers") -> list[type[Indexer]]:
        """Scan directory for Indexer subclass implementations.

        Args:
            indexer_dir: Directory to scan for indexer modules.

        Returns:
            List of Indexer subclass types found.
        """
        indexer_dir = Path(indexer_dir)
        if not indexer_dir.exists():
            return []

        indexers: list[type[Indexer]] = []

        for py_file in indexer_dir.glob("**/*.py"):
            if py_file.name.startswith("_"):
                continue

            relative_path = py_file.relative_to(indexer_dir)
            module_parts = relative_path.with_suffix("").parts
            module_name = "src.indexers." + ".".join(module_parts)
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, cls) and obj is not cls and not inspect.isabstract(obj):
                    indexers.append(obj)

        return indexers

    @classmethod
    def process_with_workers(
        cls,
        data_to_process: list[T],
        fetch: Callable[[T], U],
        callback: Callable[[U, tqdm], None],
        max_workers: int,
        description: str,
    ):
        MAX_PENDING = max_workers * 2
        pending = set()
        tickers_iter = iter(data_to_process)
        pbar = tqdm(total=len(data_to_process), desc=description)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit initial futures
            for _ in range(min(MAX_PENDING, len(data_to_process))):
                ticker = next(tickers_iter)
                future = executor.submit(fetch, ticker)
                pending.add(future)

            while pending:
                # Wait for at least one future to complete
                try:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                except:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

                # Submit new futures to replace the completed ones
                for _ in range(len(done)):
                    try:
                        ticker = next(tickers_iter)
                        pending.add(executor.submit(fetch, ticker))
                    except StopIteration:
                        break

                # Process the result
                for future in done:
                    callback(future.result(), pbar)
                    pbar.update(1)

        pbar.close()
