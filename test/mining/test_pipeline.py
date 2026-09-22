"""Tests for causalatee.mining.Pipeline: batching, concurrency, and per-stage executor isolation. Async bodies are
driven via asyncio.run() from plain sync test methods, matching this repo's existing test style, rather than
depending on an undeclared pytest-asyncio/anyio-pytest-plugin dependency for something this simple."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from causalatee.mining import Document, Pipeline, causal_predicate


class ConcurrencyTracker:
    """Tracks the maximum number of simultaneously in-flight calls, independent of wall-clock timing -- avoids flaky
    timing-based concurrency assertions."""

    def __init__(self) -> None:
        self.current = 0
        self.max_seen = 0
        self._lock = threading.Lock()

    def enter(self) -> None:
        with self._lock:
            self.current += 1
            self.max_seen = max(self.max_seen, self.current)

    def exit(self) -> None:
        with self._lock:
            self.current -= 1


async def _source(n: int):
    for i in range(n):
        yield i


def run(coro):
    return asyncio.run(coro)


class TestMap:
    def test_unbatched_sync_fn_serializes_at_concurrency_one(self):
        tracker = ConcurrencyTracker()

        def slow(x):
            tracker.enter()
            time.sleep(0.05)
            tracker.exit()
            return x * 2

        async def go():
            results = []
            await Pipeline(_source(4)).map(slow, concurrency=1).reduce(lambda x: results.append(x))
            return results

        results = run(go())
        assert sorted(results) == [0, 2, 4, 6]
        assert tracker.max_seen == 1

    def test_unbatched_sync_fn_overlaps_at_higher_concurrency(self):
        tracker = ConcurrencyTracker()

        def slow(x):
            tracker.enter()
            time.sleep(0.05)
            tracker.exit()
            return x * 2

        async def go():
            results = []
            await Pipeline(_source(4)).map(slow, concurrency=4).reduce(lambda x: results.append(x))
            return results

        results = run(go())
        assert sorted(results) == [0, 2, 4, 6]
        assert tracker.max_seen > 1

    def test_batched_fn_receives_and_returns_full_batch(self):
        calls: list[list[int]] = []

        def batch_fn(xs):
            calls.append(list(xs))
            return [x * 2 for x in xs]

        async def go():
            results = []
            await Pipeline(_source(5)).map(batch_fn, batch_size=2, concurrency=1).reduce(lambda x: results.append(x))
            return results

        results = run(go())
        assert results == [0, 2, 4, 6, 8]
        assert calls == [[0, 1], [2, 3], [4]]  # last batch is a partial group of 1

    def test_native_async_fn_is_used_directly_without_creating_an_executor(self):
        async def async_double(x):
            await asyncio.sleep(0.01)
            return x * 2

        async def go():
            pipeline = Pipeline(_source(3)).map(async_double, concurrency=2)
            results = []
            await pipeline.reduce(lambda x: results.append(x))
            return pipeline._executors

        executors = run(go())
        assert executors == []  # no thread pool needed for a genuinely async fn

    def test_exception_in_batch_propagates(self):
        def maybe_fail(x):
            if x == 2:
                raise ValueError("boom")
            return x

        async def go():
            await Pipeline(_source(4)).map(maybe_fail, concurrency=4).reduce(lambda x: None)

        with pytest.raises(ValueError, match="boom"):
            run(go())


class TestFilter:
    def test_unbatched_predicate(self):
        async def go():
            results = []
            await Pipeline(_source(6)).filter(lambda x: x % 2 == 0).reduce(lambda x: results.append(x))
            return results

        assert run(go()) == [0, 2, 4]

    def test_batched_with_custom_predicate(self):
        def batch_is_even(xs):
            return [x % 2 == 0 for x in xs]

        async def go():
            results = []
            await (
                Pipeline(_source(6))
                .filter(batch_is_even, batch_size=3, predicate=lambda r: r)
                .reduce(lambda x: results.append(x))
            )
            return results

        assert run(go()) == [0, 2, 4]

    def test_causal_predicate_helper_matches_detection_shape(self):
        assert causal_predicate({"label": "Causal", "score": 0.9}) is True
        assert causal_predicate({"label": "uncausal", "score": 0.9}) is False
        assert causal_predicate({"label": "Countercausal", "score": 0.9}) is True


class TestFlatMap:
    def test_flattens_variable_length_outputs(self):
        def split(x):
            return [x] * x  # 0 -> [], 1 -> [1], 2 -> [2, 2], 3 -> [3, 3, 3]

        async def go():
            results = []
            await Pipeline(_source(4)).flat_map(split).reduce(lambda x: results.append(x))
            return results

        assert run(go()) == [1, 2, 2, 3, 3, 3]


class TestPerStageExecutorIsolation:
    def test_two_stages_with_different_concurrency_do_not_share_an_executor(self):
        """Regression test for the shared-thread-pool bug: asyncio.to_thread's bare process-wide default pool would
        let one stage's concurrency setting bleed into another's. Each stage must own its own dedicated
        executor."""

        io_tracker = ConcurrencyTracker()
        gpu_tracker = ConcurrencyTracker()

        def io_stage(x):
            io_tracker.enter()
            time.sleep(0.05)
            io_tracker.exit()
            return x

        def gpu_stage(x):
            gpu_tracker.enter()
            time.sleep(0.05)
            gpu_tracker.exit()
            return x

        async def go():
            results = []
            await (
                Pipeline(_source(8))
                .map(io_stage, concurrency=8)
                .map(gpu_stage, concurrency=1)
                .reduce(lambda x: results.append(x))
            )
            return results

        results = run(go())
        assert sorted(results) == list(range(8))
        assert io_tracker.max_seen > 1  # the I/O stage actually got to overlap...
        assert gpu_tracker.max_seen == 1  # ...independent of the GPU stage staying serialized


class TestReduce:
    def test_returns_the_sink(self):
        async def go():
            sink_calls = []

            def sink(item):
                sink_calls.append(item)

            returned = await Pipeline(_source(3)).reduce(sink)
            return returned, sink_calls

        returned_sink, calls = run(go())
        assert calls == [0, 1, 2]
        assert callable(returned_sink)

    def test_async_sink_is_awaited(self):
        async def go():
            sink_calls = []

            async def sink(item):
                await asyncio.sleep(0.001)
                sink_calls.append(item)

            await Pipeline(_source(3)).reduce(sink)
            return sink_calls

        assert run(go()) == [0, 1, 2]


class TestMetadataPropagation:
    """map/filter/flat_map stay metadata-blind: fn only ever sees .value, and metadata is preserved (or, for
    flat_map, inherited) automatically without the model needing to know PipelineItem exists."""

    def test_bare_pipeline_items_start_with_empty_metadata(self):
        async def go():
            items = []
            await Pipeline(_source(2)).reduce_items(lambda item: items.append(item))
            return items

        items = run(go())
        assert [i.metadata for i in items] == [{}, {}]

    def test_map_preserves_metadata_unchanged(self):
        async def go():
            items = []
            await (
                Pipeline(_source(1))
                .map_metadata(lambda m: {"tag": "seed"})
                .map(lambda x: x * 10)
                .reduce_items(lambda item: items.append(item))
            )
            return items

        (item,) = run(go())
        assert item.value == 0
        assert item.metadata == {"tag": "seed"}

    def test_filter_preserves_metadata_on_kept_items(self):
        async def go():
            items = []
            await (
                Pipeline(_source(4))
                .map_metadata(lambda m: {"seen": True})
                .filter(lambda x: x % 2 == 0)
                .reduce_items(lambda item: items.append(item))
            )
            return items

        items = run(go())
        assert [i.value for i in items] == [0, 2]
        assert all(i.metadata == {"seen": True} for i in items)

    def test_flat_map_inherits_parent_metadata(self):
        async def go():
            items = []
            await (
                Pipeline(_source(2))
                .map_metadata(lambda m: {"parent": True})
                .flat_map(lambda x: [x, x + 100])
                .reduce_items(lambda item: items.append(item))
            )
            return items

        items = run(go())
        assert [i.value for i in items] == [0, 100, 1, 101]
        assert all(i.metadata == {"parent": True} for i in items)

    def test_batched_map_preserves_metadata_per_position(self):
        async def go():
            items = []
            await (
                Pipeline(_source(3))
                .map_metadata(lambda m: {})
                .annotate("doubled_of", lambda x: x)
                .map(lambda xs: [x * 2 for x in xs], batch_size=2)
                .reduce_items(lambda item: items.append(item))
            )
            return items

        items = run(go())
        assert [i.value for i in items] == [0, 2, 4]
        assert [i.metadata["doubled_of"] for i in items] == [0, 1, 2]

    def test_batched_map_mismatched_result_count_raises(self):
        def drops_one(xs):
            return [x * 2 for x in xs][:-1]  # returns one fewer result than inputs

        async def go():
            await Pipeline(_source(4)).map(drops_one, batch_size=2).reduce(lambda x: None)

        with pytest.raises(ValueError):
            run(go())


class TestAnnotate:
    def test_computes_from_value_and_leaves_value_unchanged(self):
        async def go():
            items = []
            await Pipeline(_source(3)).annotate("squared", lambda x: x * x).reduce_items(
                lambda item: items.append(item)
            )
            return items

        items = run(go())
        assert [i.value for i in items] == [0, 1, 2]
        assert [i.metadata["squared"] for i in items] == [0, 1, 4]

    def test_chained_annotate_calls_accumulate(self):
        async def go():
            items = []
            await (
                Pipeline(_source(1))
                .annotate("a", lambda x: 1)
                .annotate("b", lambda x: 2)
                .reduce_items(lambda item: items.append(item))
            )
            return items

        (item,) = run(go())
        assert item.metadata == {"a": 1, "b": 2}


class TestAnnotateWithMetadata:
    def test_fn_receives_both_value_and_current_metadata(self):
        async def go():
            items = []
            await (
                Pipeline(_source(1))
                .annotate("base", lambda x: 100)
                .annotate_with_metadata("total", lambda x, metadata: x + metadata["base"])
                .reduce_items(lambda item: items.append(item))
            )
            return items

        (item,) = run(go())
        assert item.metadata == {"base": 100, "total": 100}


class TestMapMetadata:
    def test_replaces_metadata_entirely(self):
        async def go():
            items = []
            await (
                Pipeline(_source(1))
                .annotate("a", lambda x: 1)
                .annotate("b", lambda x: 2)
                .map_metadata(lambda m: {"only": m["a"]})
                .reduce_items(lambda item: items.append(item))
            )
            return items

        (item,) = run(go())
        assert item.metadata == {"only": 1}  # "b" was dropped -- a full replace, not a merge


class TestFlatMapAnnotateKwarg:
    def test_annotate_kwarg_merges_on_top_of_inherited_metadata(self):
        def split_into_two(x):
            return [f"{x}-a", f"{x}-b"]

        async def go():
            items = []
            await (
                Pipeline(_source(1))
                .annotate("doc_id", lambda x: "doc0")
                .flat_map(split_into_two, annotate=lambda value, index: {"index": index})
                .reduce_items(lambda item: items.append(item))
            )
            return items

        items = run(go())
        assert [i.value for i in items] == ["0-a", "0-b"]
        assert [i.metadata for i in items] == [
            {"doc_id": "doc0", "index": 0},
            {"doc_id": "doc0", "index": 1},
        ]


class TestReduceItems:
    def test_reduce_hands_sink_bare_values(self):
        async def go():
            values = []
            await Pipeline(_source(3)).annotate("x", lambda x: 1).reduce(lambda v: values.append(v))
            return values

        assert run(go()) == [0, 1, 2]

    def test_reduce_items_hands_sink_full_pipeline_items(self):
        async def go():
            items = []
            await Pipeline(_source(2)).annotate("x", lambda x: 1).reduce_items(lambda item: items.append(item))
            return items

        items = run(go())
        assert [i.value for i in items] == [0, 1]
        assert [i.metadata for i in items] == [{"x": 1}, {"x": 1}]


class TestDocuments:
    def test_documents_adapts_id_and_metadata_into_a_source_key(self):
        async def source():
            yield Document(id="doc-1", text="hello", metadata={"url": "https://example.com"})
            yield Document(id="doc-2", text="world", metadata={"url": "https://example.org"})

        async def go():
            items = []
            await Pipeline.documents(source()).reduce_items(lambda item: items.append(item))
            return items

        items = run(go())
        assert [i.value for i in items] == ["hello", "world"]
        assert items[0].metadata == {"source": {"id": "doc-1", "url": "https://example.com"}}
        assert items[1].metadata == {"source": {"id": "doc-2", "url": "https://example.org"}}
