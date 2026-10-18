import sys
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


def run_parallel(
    items: Iterable[Any],
    work: Callable[[Any], Any],
    on_result: Callable[[Any, Any], None],
    workers: int,
    describe: Callable[[Any], str] = str,
) -> int:
    """Run `work` over items on a thread pool and hand each result to `on_result` on the calling
    thread as soon as it finishes, so callers can append it to disk before the next one lands.
    A crash or Ctrl-C loses only in-flight calls. Returns the number of failed items."""
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
            except Exception as e:  # noqa: BLE001 - one bad API call must not abort a paid batch
                failures += 1
                print(f"  {describe(item)}: FAILED - {e}", file=sys.stderr)
                continue
            on_result(item, result)
    return failures
