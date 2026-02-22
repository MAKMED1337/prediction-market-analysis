"""Indexer for Polymarket trades from the Polygon blockchain."""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.common.client import RateLimiter
from src.common.indexer import Indexer
from src.indexers.polymarket.blockchain import (
    CTF_EXCHANGE,
    NEGRISK_CTF_EXCHANGE,
    POLYMARKET_START_BLOCK,
    PolygonClient,
)

DATA_DIR = Path("data/polymarket/trades")
CURSOR_FILE = Path("data/polymarket/.backfill_block_cursor")


class PolymarketTradesIndexer(Indexer):
    """Fetches and stores Polymarket trades from the Polygon blockchain."""

    def __init__(
        self,
        from_block: int | None = None,
        to_block: int | None = None,
        chunk_size: int = 1_00,
        max_workers: int = 100,
        rate_limiter: RateLimiter | None = None,
    ):
        if not rate_limiter:
            rate_limiter = RateLimiter(100)

        super().__init__(
            name="polymarket_trades",
            description="Backfills Polymarket trades from Polygon blockchain to parquet files",
        )
        self._from_block = from_block
        self._to_block = to_block
        self._chunk_size = chunk_size
        self._max_workers = max_workers
        self._rate_limiter = rate_limiter

    def run(self) -> None:
        """Backfill all Polymarket trades from the Polygon blockchain.

        This fetches OrderFilled events from both CTF Exchange contracts
        (regular and NegRisk) and saves them to parquet files.
        """
        BATCH_SIZE = 10000
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)

        client = PolygonClient(rate_limiter=self._rate_limiter)
        current_block = client.get_block_number()

        # Determine starting block
        from_block = self._from_block
        if from_block is None:
            if CURSOR_FILE.exists():
                try:
                    from_block = int(CURSOR_FILE.read_text().strip())
                    print(f"Resuming from block {from_block}")
                except (ValueError, TypeError):
                    from_block = POLYMARKET_START_BLOCK
            else:
                from_block = POLYMARKET_START_BLOCK

        to_block = self._to_block
        if to_block is None:
            to_block = current_block

        print(f"Fetching trades from block {from_block} to {to_block}")
        print(f"Total blocks: {to_block - from_block:,}")

        all_trades = []
        total_saved = 0
        contracts = [
            ("CTF Exchange", CTF_EXCHANGE),
            ("NegRisk CTF Exchange", NEGRISK_CTF_EXCHANGE),
        ]

        def get_next_chunk_idx():
            existing = list(DATA_DIR.glob("trades_*.parquet"))
            if not existing:
                return 0
            indices = []
            for f in existing:
                parts = f.stem.split("_")
                if len(parts) >= 2:
                    try:
                        indices.append(int(parts[1]))
                    except ValueError:
                        pass
            return max(indices) + BATCH_SIZE if indices else 0

        chunk_idx = get_next_chunk_idx()

        def save_batch(trades_batch):
            nonlocal total_saved, chunk_idx
            if not trades_batch:
                return
            chunk_path = DATA_DIR / f"trades_{chunk_idx}_{chunk_idx + BATCH_SIZE}.parquet"
            chunk_idx += BATCH_SIZE

            df = pd.DataFrame(trades_batch)
            df.to_parquet(chunk_path)
            total_saved += len(trades_batch)
            tqdm.write(f"Saved {len(trades_batch)} trades to {chunk_path.name}")

        def fetch_range(chunk: tuple[int, int]) -> tuple[tuple[int, int], list[dict]]:
            fetched_at = datetime.utcnow()
            local_trades = []

            for contract_name, contract_address in contracts:
                trades = client.get_trades(
                    from_block=chunk[0],
                    to_block=chunk[1],
                    contract_address=contract_address,
                )

                for trade in trades:
                    trade_dict = asdict(trade)
                    # Convert large ints to strings to avoid parquet overflow
                    trade_dict["maker_asset_id"] = str(trade_dict["maker_asset_id"])
                    trade_dict["taker_asset_id"] = str(trade_dict["taker_asset_id"])
                    trade_dict["_fetched_at"] = fetched_at
                    trade_dict["_contract"] = contract_name
                    local_trades.append(trade_dict)
            return chunk, local_trades

        # Build list of chunk ranges
        ranges = []
        current = from_block
        while current <= to_block:
            end = min(current + self._chunk_size - 1, to_block)
            ranges.append((current, end))
            current = end + 1

        range_ptr = 0
        waiting_to_save = {}

        def save_progress(limit: int = BATCH_SIZE) -> None:
            nonlocal all_trades, waiting_to_save, range_ptr

            while True:
                if range_ptr >= len(ranges):
                    assert len(waiting_to_save) == 0
                    break

                range_ = ranges[range_ptr]
                if range_ not in waiting_to_save:
                    break
                all_trades.extend(waiting_to_save.pop(range_))
                range_ptr += 1

            while len(all_trades) >= limit:
                # In the end (non full blocks as well) we will store continues chunk up to last range
                assert range_ptr > 0
                CURSOR_FILE.write_text(str(ranges[range_ptr - 1][1]))
                save_batch(all_trades[:BATCH_SIZE])
                all_trades = all_trades[BATCH_SIZE:]

        def process_result(result: tuple[tuple[int, int], list[dict]], pbar: tqdm) -> None:
            nonlocal all_trades, waiting_to_save, range_ptr

            chunk, blocks = result
            waiting_to_save[chunk] = blocks
            save_progress()

            pbar.set_postfix(
                block=ranges[range_ptr][1] if range_ptr < len(ranges) else "finished",
                buffer=len(all_trades),
                waiting_to_save=len(waiting_to_save),
                saved=total_saved,
            )

        try:
            self.process_with_workers(
                ranges,
                fetch_range,
                process_result,
                self._max_workers,
                "Backfilling polymarket trades",
            )
        finally:
            save_progress(1)

        # if CURSOR_FILE.exists():
        #     CURSOR_FILE.unlink()

        print(f"\nBackfill complete: {total_saved} trades saved")
