"""A fluent, streaming, concurrency-aware async pipeline over an async source.

Built on ``aiostream`` for exactly one hard primitive it gets right -- bounded-concurrency async mapping with correct
backpressure, ordering, and exception propagation (``aiostream.stream.map(..., task_limit=N)``) -- and plain Python
async generators for everything else (batching/unbatching lists), so the "black box" surface stays as small as possible.

Central, verified fact this module is built around: passing a *synchronous* function to ``aiostream.stream.map`` runs it
INLINE on the event loop (blocks everything else while it runs), and ``task_limit`` is REJECTED outright for sync
functions ("can only be used when the provided function is asynchronous"). So any stage wanting concurrency control --
including every ``causalatee.models``-based stage, since those protocols are always plain sync callables -- MUST be
wrapped as an async function. ``Pipeline`` does this wrapping automatically via ``loop.run_in_executor``, using a
DEDICATED ``ThreadPoolExecutor`` per stage sized to that stage's own ``concurrency`` -- not the shared process-wide
default pool ``asyncio.to_thread`` uses, which would let an unrelated high-concurrency I/O stage silently starve a
``concurrency=1`` GPU stage (or vice versa) regardless of ``task_limit``.

In-flight ``run_in_executor`` calls are not cancellable -- an accepted limitation, not something this module tries to
fix.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Generic, TypeVar

from aiostream import stream

T = TypeVar("T")
R = TypeVar("R")

AsyncOrSyncFn = Callable[[Any], Any]


async def _in_batches(source: AsyncIterable[T], batch_size: int) -> AsyncIterator[list[T]]:
    batch: list[T] = []
    async for item in source:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class Pipeline(Generic[T]):
    """A chainable, lazily-evaluated stream of items of type ``T``.

    Nothing runs until :meth:`reduce` (or manual ``async for``) is awaited -- every stage is composed by wrapping
    async generators, never materialized in between.
    """

    def __init__(self, source: AsyncIterable[T], _executors: list[ThreadPoolExecutor] | None = None) -> None:
        self._source = source
        self._executors: list[ThreadPoolExecutor] = _executors if _executors is not None else []

    def __aiter__(self) -> AsyncIterator[T]:
        return self._source.__aiter__()

    def _as_async(self, fn: AsyncOrSyncFn, concurrency: int) -> Callable[[Any], Awaitable[Any]]:
        """Return an async callable equivalent to ``fn``, owning a dedicated executor (tracked for cleanup in
        :meth:`reduce`) if ``fn`` was synchronous."""

        if asyncio.iscoroutinefunction(fn):
            return fn

        executor = ThreadPoolExecutor(max_workers=max(concurrency, 1))
        self._executors.append(executor)

        async def wrapped(arg: Any) -> Any:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(executor, fn, arg)

        return wrapped

    def map(
        self,
        fn: AsyncOrSyncFn,
        *,
        concurrency: int = 1,
        batch_size: int = 1,
    ) -> Pipeline[R]:
        """Apply ``fn`` to every item.

        At ``batch_size=1`` (default), ``fn`` is called once per item, naturally (``fn(item) -> result``) -- the
        ordinary single-item mapping convention.

        At ``batch_size > 1``, ``fn`` is called once per group of up to ``batch_size`` items, and must accept
        and return a list of the SAME length (``fn(items: list[T]) -> list[R]``). ``causalatee.models`` protocol
        instances are already batch-native this way, so plugging one in as ``fn`` with ``batch_size > 1`` needs
        no extra glue.

        ``concurrency`` bounds how many calls to ``fn`` may be in flight at once. Use ``concurrency=1`` for a
        single GPU model (its own internal batching, via ``batch_size``, is what gives it throughput --
        concurrent calls would just contend for the same device). Use a higher ``concurrency`` for I/O-bound
        work.
        """

        async_fn = self._as_async(fn, concurrency)

        if batch_size == 1:
            # aiostream's stubs model map's fn as accepting *args (for multi-source zipping we don't use);
            # verified correct at runtime, see module probe.
            mapped: Any = stream.map(self._source, async_fn, task_limit=concurrency)  # type: ignore[arg-type]

            async def one_at_a_time() -> AsyncIterator[R]:
                async with mapped.stream() as streamer:
                    async for result in streamer:
                        yield result

            return Pipeline(one_at_a_time(), self._executors)

        batches = _in_batches(self._source, batch_size)
        # aiostream's stubs model map's fn as accepting *args (for multi-source zipping we don't use);
        # verified correct at runtime, see module probe.
        mapped_batches: Any = stream.map(batches, async_fn, task_limit=concurrency)  # type: ignore[arg-type]

        async def unbatched() -> AsyncIterator[R]:
            async with mapped_batches.stream() as streamer:
                async for result_batch in streamer:
                    for item in result_batch:
                        yield item

        return Pipeline(unbatched(), self._executors)

    def filter(
        self,
        fn: AsyncOrSyncFn,
        *,
        predicate: Callable[[Any], bool] = bool,
        concurrency: int = 1,
        batch_size: int = 1,
    ) -> Pipeline[T]:
        """Keep only items for which ``predicate(fn(item))`` is true (or, batched, ``predicate(fn(batch)[i])`` for
        each item in the batch).

        ``fn`` need not itself return a boolean -- e.g. plug a ``causalatee.models.Detection`` model in directly
        as ``fn`` with ``predicate=causal_predicate`` (see this package's ``causal_predicate``) rather than
        writing that translation yourself each time.
        """

        async_fn = self._as_async(fn, concurrency)

        if batch_size == 1:

            async def process_one(item: T) -> tuple[T, Any]:
                return item, await async_fn(item)

            # aiostream's stubs model map's fn as accepting *args (for multi-source zipping we don't use);
            # verified correct at runtime, see module probe.
            mapped: Any = stream.map(self._source, process_one, task_limit=concurrency)  # type: ignore[arg-type]

            async def filtered_one() -> AsyncIterator[T]:
                async with mapped.stream() as streamer:
                    async for item, result in streamer:
                        if predicate(result):
                            yield item

            return Pipeline(filtered_one(), self._executors)

        batches = _in_batches(self._source, batch_size)

        async def process_batch(batch: list[T]) -> list[tuple[T, Any]]:
            results = await async_fn(batch)
            return list(zip(batch, results))

        # aiostream's stubs model map's fn as accepting *args (for multi-source zipping we don't use);
        # verified correct at runtime, see module probe.
        mapped_batches: Any = stream.map(batches, process_batch, task_limit=concurrency)  # type: ignore[arg-type]

        async def filtered_batched() -> AsyncIterator[T]:
            async with mapped_batches.stream() as streamer:
                async for pairs in streamer:
                    for item, result in pairs:
                        if predicate(result):
                            yield item

        return Pipeline(filtered_batched(), self._executors)

    def flat_map(
        self,
        fn: AsyncOrSyncFn,
        *,
        concurrency: int = 1,
    ) -> Pipeline[R]:
        """Apply ``fn`` to every item, where ``fn(item) -> Iterable[R]`` returns zero or more output items per
        input (e.g. splitting a document into sentences), and flatten the results into a single stream of ``R``."""

        async_fn = self._as_async(fn, concurrency)
        # aiostream's stubs model map's fn as accepting *args (for multi-source zipping we don't use);
        # verified correct at runtime, see module probe.
        mapped: Any = stream.map(self._source, async_fn, task_limit=concurrency)  # type: ignore[arg-type]

        async def flattened() -> AsyncIterator[R]:
            async with mapped.stream() as streamer:
                async for results in streamer:
                    for item in results:
                        yield item

        return Pipeline(flattened(), self._executors)

    async def reduce(self, sink: Callable[[T], Awaitable[None] | None]) -> Any:
        """Drain the pipeline, calling ``sink(item)`` (sync or async) for every item, then shut down every executor
        any stage in this chain created, and return ``sink`` itself (most sinks, e.g. this package's graph sink,
        expose their accumulated result as an attribute/method after draining)."""

        try:
            async for item in self:
                result = sink(item)
                if asyncio.iscoroutine(result):
                    await result
        finally:
            for executor in self._executors:
                executor.shutdown()
        return sink


def causal_predicate(result: Any) -> bool:
    """``predicate=`` for :meth:`Pipeline.filter` when ``fn`` is a ``causalatee.models.Detection`` model directly: true
    unless labelled "Uncausal" (case-insensitive) -- the exact check ``causalatee.models._ComposedExtraction`` makes
    internally, so a mining pipeline's detection-gating stage agrees with the end-to-end ``Extraction`` composition
    rather than drifting from it."""

    return str(result["label"]).lower() != "uncausal"
