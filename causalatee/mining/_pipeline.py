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

Provenance/metadata design: every item flowing through a ``Pipeline`` is internally a ``PipelineItem`` (a value paired
with a free-form metadata mapping). ``map``/``filter``/``flat_map`` deliberately stay metadata-BLIND -- they unwrap
``.value`` before calling the user's function and rewrap the result with the SAME (or, for ``flat_map``, inherited)
metadata automatically, so ordinary ``causalatee.models`` Protocol implementations (plain ``T -> R`` functions) never
need to know ``PipelineItem`` exists. ``annotate``/``annotate_with_metadata``/``map_metadata`` are the explicit,
opt-in operations for reading or changing metadata. ``reduce``/``reduce_items`` mirror this split at the drain end:
``reduce`` hands a sink plain values (the ergonomic default every existing sink already expects), ``reduce_items``
hands a metadata-aware sink (e.g. ``causalatee.mining.graph_sink``) the full ``PipelineItem``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from aiostream import stream

from ._source import DocumentSource

T = TypeVar("T")
R = TypeVar("R")

AsyncOrSyncFn = Callable[..., Any]


@dataclass(frozen=True)
class PipelineItem(Generic[T]):
    """One value flowing through a ``Pipeline``, paired with its accumulated provenance/metadata.

    ``metadata`` is intentionally generic and source-dependent (a source URL, a WARC record id, a derived sentence
    index or topic label, ...) -- ``Pipeline`` itself has no opinion on its shape, only on how it propagates.
    """

    value: T
    metadata: Mapping[str, object] = field(default_factory=dict)


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

    Nothing runs until :meth:`reduce`/:meth:`reduce_items` (or manual ``async for``) is awaited -- every stage is
    composed by wrapping async generators, never materialized in between.
    """

    def __init__(self, source: AsyncIterable[T], _executors: list[ThreadPoolExecutor] | None = None) -> None:
        """Wrap a plain ``AsyncIterable[T]`` of bare values -- each is paired with empty metadata. Use
        :meth:`documents` instead to build a pipeline whose items already carry source provenance."""

        async def wrapped() -> AsyncIterator[PipelineItem[T]]:
            async for value in source:
                yield PipelineItem(value)

        self._source: AsyncIterable[PipelineItem[T]] = wrapped()
        self._executors: list[ThreadPoolExecutor] = _executors if _executors is not None else []

    @staticmethod
    def _from_items(source: AsyncIterable[PipelineItem[R]], executors: list[ThreadPoolExecutor]) -> Pipeline[R]:
        """Internal constructor: wrap an already-``PipelineItem``-yielding source directly, with no additional
        wrapping -- every chaining method (``map``/``filter``/``flat_map``/``annotate``/...) returns through this,
        never through ``__init__``, so items are never double-wrapped. A plain ``@staticmethod`` (not
        ``@classmethod``) deliberately -- ``Pipeline`` isn't meant to be subclassed, and this sidesteps a known
        mypy limitation inferring a fresh generic parameter through ``cls.__new__(cls)``."""

        self: Pipeline[R] = Pipeline.__new__(Pipeline)
        self._source = source
        self._executors = executors
        return self

    @classmethod
    def documents(cls, source: DocumentSource) -> Pipeline[str]:
        """Adapt a ``DocumentSource`` into a ``Pipeline[str]``: each ``Document``'s ``.text`` becomes the item's
        ``value``, and its ``.id``/``.metadata`` are folded into a ``"source"`` sub-key of the item's metadata --
        so extraction/detection models keep operating on plain strings (matching every ``causalatee.models``
        Protocol) while provenance flows automatically from the very first stage."""

        async def wrapped() -> AsyncIterator[PipelineItem[str]]:
            async for document in source:
                yield PipelineItem(document.text, {"source": {"id": document.id, **document.metadata}})

        return cls._from_items(wrapped(), [])

    def __aiter__(self) -> AsyncIterator[T]:
        async def values() -> AsyncIterator[T]:
            async for item in self._source:
                yield item.value

        return values()

    def _as_async(self, fn: AsyncOrSyncFn, concurrency: int) -> Callable[..., Awaitable[Any]]:
        """Return an async callable equivalent to ``fn``, owning a dedicated executor (tracked for cleanup in
        :meth:`reduce`/:meth:`reduce_items`) if ``fn`` was synchronous. Forwards any number of positional args
        through to ``fn`` (unary for ``map``/``filter``/``flat_map``/``annotate``, binary for
        ``annotate_with_metadata``, which also needs the item's current metadata)."""

        if asyncio.iscoroutinefunction(fn):
            return fn

        executor = ThreadPoolExecutor(max_workers=max(concurrency, 1))
        self._executors.append(executor)

        async def wrapped(*args: Any) -> Any:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(executor, fn, *args)

        return wrapped

    def map(
        self,
        fn: AsyncOrSyncFn,
        *,
        concurrency: int = 1,
        batch_size: int = 1,
    ) -> Pipeline[R]:
        """Apply ``fn`` to every item's ``value``, preserving its metadata unchanged.

        At ``batch_size=1`` (default), ``fn`` is called once per item, naturally (``fn(value) -> result``) -- the
        ordinary single-item mapping convention.

        At ``batch_size > 1``, ``fn`` is called once per group of up to ``batch_size`` values, and must accept
        and return a list of the SAME length (``fn(values: list[T]) -> list[R]``) -- enforced via ``zip(...,
        strict=True)`` when results are paired back with their original items, so a model that silently drops or
        duplicates a result raises immediately instead of misattributing provenance. ``causalatee.models`` protocol
        instances are already batch-native this way, so plugging one in as ``fn`` with ``batch_size > 1`` needs
        no extra glue.

        ``concurrency`` bounds how many calls to ``fn`` may be in flight at once. Use ``concurrency=1`` for a
        single GPU model (its own internal batching, via ``batch_size``, is what gives it throughput --
        concurrent calls would just contend for the same device). Use a higher ``concurrency`` for I/O-bound
        work.
        """

        async_fn = self._as_async(fn, concurrency)

        if batch_size == 1:

            async def item_fn(item: PipelineItem[T]) -> PipelineItem[R]:
                result = await async_fn(item.value)
                return PipelineItem(result, item.metadata)

            # aiostream's stubs model map's fn as accepting *args (for multi-source zipping we don't use);
            # verified correct at runtime, see module probe.
            mapped: Any = stream.map(self._source, item_fn, task_limit=concurrency)  # type: ignore[arg-type]

            async def one_at_a_time() -> AsyncIterator[PipelineItem[R]]:
                async with mapped.stream() as streamer:
                    async for result in streamer:
                        yield result

            return Pipeline._from_items(one_at_a_time(), self._executors)

        batches = _in_batches(self._source, batch_size)

        async def batch_fn(batch: list[PipelineItem[T]]) -> list[PipelineItem[R]]:
            results = await async_fn([item.value for item in batch])
            return [PipelineItem(result, item.metadata) for item, result in zip(batch, results, strict=True)]

        # aiostream's stubs model map's fn as accepting *args (for multi-source zipping we don't use);
        # verified correct at runtime, see module probe.
        mapped_batches: Any = stream.map(batches, batch_fn, task_limit=concurrency)  # type: ignore[arg-type]

        async def unbatched() -> AsyncIterator[PipelineItem[R]]:
            async with mapped_batches.stream() as streamer:
                async for result_batch in streamer:
                    for item in result_batch:
                        yield item

        return Pipeline._from_items(unbatched(), self._executors)

    def filter(
        self,
        fn: AsyncOrSyncFn,
        *,
        predicate: Callable[[Any], bool] = bool,
        concurrency: int = 1,
        batch_size: int = 1,
    ) -> Pipeline[T]:
        """Keep only items for which ``predicate(fn(value))`` is true (or, batched, ``predicate(fn(values)[i])``
        for each value in the batch), preserving metadata on every kept item unchanged.

        ``fn`` need not itself return a boolean -- e.g. plug a ``causalatee.models.Detection`` model in directly
        as ``fn`` with ``predicate=causal_predicate`` (see this package's ``causal_predicate``) rather than
        writing that translation yourself each time.
        """

        async_fn = self._as_async(fn, concurrency)

        if batch_size == 1:

            async def process_one(item: PipelineItem[T]) -> tuple[PipelineItem[T], Any]:
                return item, await async_fn(item.value)

            # aiostream's stubs model map's fn as accepting *args (for multi-source zipping we don't use);
            # verified correct at runtime, see module probe.
            mapped: Any = stream.map(self._source, process_one, task_limit=concurrency)  # type: ignore[arg-type]

            async def filtered_one() -> AsyncIterator[PipelineItem[T]]:
                async with mapped.stream() as streamer:
                    async for item, result in streamer:
                        if predicate(result):
                            yield item

            return Pipeline._from_items(filtered_one(), self._executors)

        batches = _in_batches(self._source, batch_size)

        async def process_batch(batch: list[PipelineItem[T]]) -> list[tuple[PipelineItem[T], Any]]:
            results = await async_fn([item.value for item in batch])
            return list(zip(batch, results, strict=True))

        # aiostream's stubs model map's fn as accepting *args (for multi-source zipping we don't use);
        # verified correct at runtime, see module probe.
        mapped_batches: Any = stream.map(batches, process_batch, task_limit=concurrency)  # type: ignore[arg-type]

        async def filtered_batched() -> AsyncIterator[PipelineItem[T]]:
            async with mapped_batches.stream() as streamer:
                async for pairs in streamer:
                    for item, result in pairs:
                        if predicate(result):
                            yield item

        return Pipeline._from_items(filtered_batched(), self._executors)

    def flat_map(
        self,
        fn: AsyncOrSyncFn,
        *,
        concurrency: int = 1,
        annotate: Callable[[R, int], Mapping[str, object]] | None = None,
    ) -> Pipeline[R]:
        """Apply ``fn`` to every item's ``value``, where ``fn(value) -> Iterable[R]`` returns zero or more output
        values per input (e.g. splitting a document into sentences), and flatten the results into a single stream.

        Each resulting item inherits the parent's metadata. ``annotate``, if given, is called as
        ``annotate(child_value, index)`` for each of the (up to) N values ``fn`` produced (``index`` counting from
        0 within that one expansion) and its return value is MERGED on top of the inherited metadata -- the
        mechanism for attaching metadata the expansion itself introduces (e.g. a sentence's own text/index within
        its source document), without a separate ``.annotate()`` call per new field.
        """

        async_fn = self._as_async(fn, concurrency)

        async def item_fn(item: PipelineItem[T]) -> list[PipelineItem[R]]:
            results = await async_fn(item.value)
            children = []
            for index, value in enumerate(results):
                metadata = {**item.metadata, **annotate(value, index)} if annotate is not None else item.metadata
                children.append(PipelineItem(value, metadata))
            return children

        # aiostream's stubs model map's fn as accepting *args (for multi-source zipping we don't use);
        # verified correct at runtime, see module probe.
        mapped: Any = stream.map(self._source, item_fn, task_limit=concurrency)  # type: ignore[arg-type]

        async def flattened() -> AsyncIterator[PipelineItem[R]]:
            async with mapped.stream() as streamer:
                async for results in streamer:
                    for item in results:
                        yield item

        return Pipeline._from_items(flattened(), self._executors)

    def annotate(self, name: str, fn: AsyncOrSyncFn, *, concurrency: int = 1) -> Pipeline[T]:
        """Compute ``fn(value)`` and store it under ``metadata[name]``, leaving ``value`` itself unchanged.

        Deliberately metadata-BLIND on input, matching ``map``/``filter``/``flat_map``'s "models operate on
        values" principle -- use :meth:`annotate_with_metadata` for a derived value that itself needs the item's
        existing metadata as input.
        """

        async_fn = self._as_async(fn, concurrency)

        async def item_fn(item: PipelineItem[T]) -> PipelineItem[T]:
            computed = await async_fn(item.value)
            return PipelineItem(item.value, {**item.metadata, name: computed})

        # aiostream's stubs model map's fn as accepting *args (for multi-source zipping we don't use);
        # verified correct at runtime, see module probe.
        mapped: Any = stream.map(self._source, item_fn, task_limit=concurrency)  # type: ignore[arg-type]

        async def annotated() -> AsyncIterator[PipelineItem[T]]:
            async with mapped.stream() as streamer:
                async for result in streamer:
                    yield result

        return Pipeline._from_items(annotated(), self._executors)

    def annotate_with_metadata(
        self,
        name: str,
        fn: Callable[[T, Mapping[str, object]], Any],
        *,
        concurrency: int = 1,
    ) -> Pipeline[T]:
        """Like :meth:`annotate`, but ``fn(value, metadata)`` also receives the item's CURRENT metadata -- for a
        derived value that needs existing provenance as input (e.g. a recency score computed from
        ``metadata["source"]["timestamp"]`` together with the text itself)."""

        async_fn = self._as_async(fn, concurrency)

        async def item_fn(item: PipelineItem[T]) -> PipelineItem[T]:
            computed = await async_fn(item.value, item.metadata)
            return PipelineItem(item.value, {**item.metadata, name: computed})

        # aiostream's stubs model map's fn as accepting *args (for multi-source zipping we don't use);
        # verified correct at runtime, see module probe.
        mapped: Any = stream.map(self._source, item_fn, task_limit=concurrency)  # type: ignore[arg-type]

        async def annotated() -> AsyncIterator[PipelineItem[T]]:
            async with mapped.stream() as streamer:
                async for result in streamer:
                    yield result

        return Pipeline._from_items(annotated(), self._executors)

    def map_metadata(
        self,
        fn: Callable[[Mapping[str, object]], Mapping[str, object]],
        *,
        concurrency: int = 1,
    ) -> Pipeline[T]:
        """Replace an item's metadata with ``fn(metadata)`` -- for arbitrary restructuring (adding, removing, or
        reshaping fields), unlike ``annotate``'s always-additive single key. ``fn``'s return value REPLACES the
        metadata entirely; merge in ``**metadata`` yourself if you only mean to add to it."""

        async_fn = self._as_async(fn, concurrency)

        async def item_fn(item: PipelineItem[T]) -> PipelineItem[T]:
            new_metadata = await async_fn(item.metadata)
            return PipelineItem(item.value, new_metadata)

        # aiostream's stubs model map's fn as accepting *args (for multi-source zipping we don't use);
        # verified correct at runtime, see module probe.
        mapped: Any = stream.map(self._source, item_fn, task_limit=concurrency)  # type: ignore[arg-type]

        async def transformed() -> AsyncIterator[PipelineItem[T]]:
            async with mapped.stream() as streamer:
                async for result in streamer:
                    yield result

        return Pipeline._from_items(transformed(), self._executors)

    async def _drain(self, sink: Callable[[Any], Any], unwrap: Callable[[PipelineItem[T]], Any]) -> Any:
        try:
            async for item in self._source:
                result = sink(unwrap(item))
                if asyncio.iscoroutine(result):
                    await result
        finally:
            for executor in self._executors:
                executor.shutdown()
        return sink

    async def reduce(self, sink: Callable[[T], Awaitable[None] | None]) -> Any:
        """Drain the pipeline, calling ``sink(item.value)`` (sync or async) for every item -- metadata-BLIND, the
        ergonomic default every ordinary sink already expects. Shuts down every executor any stage in this chain
        created, then returns ``sink`` itself (most sinks, e.g. this package's graph sink, expose their
        accumulated result as an attribute/method after draining). See :meth:`reduce_items` for a sink that needs
        the full ``PipelineItem`` (value AND metadata)."""

        return await self._drain(sink, lambda item: item.value)

    async def reduce_items(self, sink: Callable[[PipelineItem[T]], Awaitable[None] | None]) -> Any:
        """Like :meth:`reduce`, but calls ``sink(item)`` with the full ``PipelineItem`` -- for a metadata-aware
        sink such as ``causalatee.mining.graph_sink``, which needs provenance to compute per-edge evidence/support.
        """

        return await self._drain(sink, lambda item: item)


def causal_predicate(result: Any) -> bool:
    """``predicate=`` for :meth:`Pipeline.filter` when ``fn`` is a ``causalatee.models.Detection`` model directly: true
    unless labeled "Uncausal" (case-insensitive) -- the exact check ``causalatee.models._ComposedExtraction`` makes
    internally, so a mining pipeline's detection-gating stage agrees with the end-to-end ``Extraction`` composition
    rather than drifting from it."""

    return str(result["label"]).lower() != "uncausal"
