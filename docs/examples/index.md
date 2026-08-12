# Examples

Runnable notebooks demonstrating `causalatee` end to end, grouped by what they cover.

## Fine-tuning Task Models

<div class="grid cards" markdown>

-   :dart: **Causality Detection**

    ---

    Fine-tune a transformer for the binary "does this sentence express a causal relation?"
    classification task.

    [Open notebook →](detection.ipynb)

-   :mag: **Causal Candidate Extraction**

    ---

    Fine-tune a span-extraction model to find every cause/effect entity span in a sentence.

    [Open notebook →](candidate_extraction.ipynb)

-   :link: **Causality Identification**

    ---

    Fine-tune a relation classifier: given two already-marked entity spans, does a causal
    relation hold between them?

    [Open notebook →](identification.ipynb)

</div>

## Baselines

<div class="grid cards" markdown>

-   :deciduous_tree: **SDP Causality Extraction**

    ---

    Extract causal relations from a syntactic dependency parse's shortest path between two
    entities — no fine-tuning required.

    [Open notebook →](sdp_causality_extraction.ipynb)

</div>

## Causal Graphs & Mining

<div class="grid cards" markdown>

-   :spider_web: **CauseNet and CGF**

    ---

    Load a [CauseNet](https://causenet.org) sample, save it as **CGF**, and load it back
    memory-mapped, without materializing it in Python objects.

    [Open notebook →](causenet_cgf.ipynb)

-   :pick: **Mining a Causal Graph**

    ---

    Turn a corpus of raw documents into an aggregated causal graph with
    `causalatee.mining`'s streaming pipeline.

    [Open notebook →](mining.ipynb)

</div>
