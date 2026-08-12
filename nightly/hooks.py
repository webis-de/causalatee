"""mkdocs-macros hooks — generate repetitive dataset page sections from YAML frontmatter."""

import shutil
from pathlib import Path


def on_pre_build(config):
    """Copy examples/ notebooks into docs/examples/ before MkDocs resolves nav paths."""
    src = Path("examples")
    dst = Path(config["docs_dir"]) / "examples"
    dst.mkdir(exist_ok=True)
    for nb in src.glob("*.ipynb"):
        shutil.copy(nb, dst / nb.name)


_TASK_NAMES = {
    "causality-detection": "Causality Detection",
    "causal-candidate-extraction": "Causal Event Candidate Extraction",
    "causality-identification": "Causality Identification",
}

_TASK_LINKS = {
    "causality-detection": "../tasks/causality_detection.md",
    "causal-candidate-extraction": "../tasks/causal_event_candidate_detection.md",
    "causality-identification": "../tasks/causality_identification.md",
}

_HF_CONFIGS = {
    "causality-detection": "causality%20detection",
    "causal-candidate-extraction": "causal%20candidate%20extraction",
    "causality-identification": "causality%20identification",
}

# A dataset page's frontmatter may set ``granularity: inter-sentence`` to
# flag that its rows are whole documents/sections rather than individual
# sentences (e.g. BioCause -- see its .md page for why). Defaults to
# "intra-sentence" (the common case) when unset, so existing dataset pages
# don't all need updating.
_GRANULARITY_LABELS = {
    "intra-sentence": "Intra-sentence",
    "inter-sentence": "Inter-sentence (whole document/section)",
}
_DEFAULT_GRANULARITY = "intra-sentence"

# A dataset page's frontmatter may set ``polarity``/``strength`` to describe whether its
# original annotation distinguishes countercausal (negated-effect) relations, or hedged/
# asserted causal strength, and whether causalatee's converter currently surfaces that in the
# converted data:
#   - "none": the original annotation carries no such signal.
#   - "documented": the signal exists in the original annotation but causalatee's converter
#     does not (yet) surface it -- usually because the upstream source causalatee actually reads
#     (e.g. a UniCausal aggregation) already stripped the field before causalatee's own code runs.
#   - "captured": the converted data actually carries the signal (e.g. via Relation.Countercausal).
# Unset on a dataset page means "not yet assessed", not "none" -- the pill is simply omitted.
_POLARITY_LABELS = {
    "none": "No Polarity Signal",
    "documented": "Polarity Documented, Not Converted",
    "captured": "Polarity Captured",
}
_STRENGTH_LABELS = {
    "none": "No Strength Signal",
    "documented": "Strength Documented, Not Converted",
    "captured": "Strength Captured",
}

# Order the "Supported Causal Graphs" accordions appear in on causalgraphs.md --
# matches the order they're introduced in that page's own prose, not alphabetical.
_GRAPH_ORDER = ["causenet", "cgf", "cause_effect_graph", "sql_graph"]


def define_env(env):
    @env.macro
    def dataset_pills():
        """Small colored ``<span class="pill pill-...">`` tags summarizing
        at-a-glance dataset properties: granularity (always shown), plus
        polarity/strength (shown only when the page's frontmatter sets them —
        see ``_POLARITY_LABELS``/``_STRENGTH_LABELS``)."""
        meta = env.page.meta
        granularity_key = meta.get("granularity", _DEFAULT_GRANULARITY)
        granularity_label = _GRANULARITY_LABELS.get(granularity_key, granularity_key)
        pills = [f'<span class="pill pill-{granularity_key}">{granularity_label}</span>']
        for field, labels in (("polarity", _POLARITY_LABELS), ("strength", _STRENGTH_LABELS)):
            key = meta.get(field)
            if key is None:
                continue
            label = labels.get(key, key)
            pills.append(f'<span class="pill pill-{field}-{key}">{label}</span>')
        return " ".join(pills) + "\n\n"

    @env.macro
    def dataset_badges():
        meta = env.page.meta
        doi = meta.get("doi")
        hf_page = meta.get("hf_page")
        parts = []
        if doi:
            parts.append(f"[:page_facing_up:](https://doi.org/{doi})")
        if hf_page:
            parts.append(f"[:hugging:]({hf_page})")
        return (" ".join(parts) + "\n\n") if parts else ""

    @env.macro
    def dataset_overview():
        meta = env.page.meta
        domain = meta.get("domain", "—")
        year = meta.get("year", "—")
        sentences = meta.get("sentences", "—")
        granularity = _GRANULARITY_LABELS.get(
            meta.get("granularity", _DEFAULT_GRANULARITY),
            meta.get("granularity"),
        )
        tasks = meta.get("supported_tasks") or {}

        lines = [
            "## Overview\n\n",
            "| | |\n|---|---|\n",
            f"| **Domain** | {domain} |\n",
            f"| **Year** | {year} |\n",
            f"| **Sentences** | {sentences} |\n",
            f"| **Granularity** | {granularity} |\n",
            "\n## Tasks & Splits\n\n",
            "| Task | Train | Test |\n|------|------:|-----:|\n",
        ]
        for task_id, task_info in tasks.items():
            name = _TASK_NAMES.get(task_id, task_id)
            link = _TASK_LINKS.get(task_id, "#")
            splits = (task_info or {}).get("splits") or {}
            train = splits.get("train", "—")
            test = splits.get("test", "—")
            train_s = f"{train:,}" if isinstance(train, int) else str(train)
            test_s = f"{test:,}" if isinstance(test, int) else str(test)
            lines.append(f"| [{name}]({link}) | {train_s} | {test_s} |\n")
        return "".join(lines) + "\n"

    @env.macro
    def dataset_viewer():
        meta = env.page.meta
        hf_repo = meta.get("hf_repo")
        tasks = meta.get("supported_tasks") or {}

        embeds = []
        for task_id, task_info in tasks.items():
            info = task_info or {}
            url = info.get("hfurl")
            if not url and hf_repo and task_id in _HF_CONFIGS:
                config = _HF_CONFIGS[task_id]
                url = f"https://huggingface.co/datasets/{hf_repo}/embed/viewer/{config}/train"
            if url:
                embeds.append((task_id, url))

        if not embeds:
            return ""

        lines = ["## Data Explorer\n\n"]
        for task_id, url in embeds:
            if len(embeds) > 1:
                lines.append(f"### {_TASK_NAMES.get(task_id, task_id)}\n\n")
            lines.append(f'<iframe src="{url}" frameborder="0" width="100%" height="560px"></iframe>\n\n')
        return "".join(lines)

    @env.macro
    def tira_leaderboard():
        meta = env.page.meta
        tasks = meta.get("supported_tasks") or {}

        entries = [
            (task_id, (task_info or {}).get("tiraurl"))
            for task_id, task_info in tasks.items()
            if (task_info or {}).get("tiraurl")
        ]
        if not entries:
            return ""

        lines = ["## Leaderboard\n\n"]
        for task_id, url in entries:
            if len(entries) > 1:
                lines.append(f"### {_TASK_NAMES.get(task_id, task_id)}\n\n")
            lines.append(
                f'<div class="tira-leaderboard" data-tira-url="{url}"><p><em>Loading leaderboard…</em></p></div>\n\n'
            )
        return "".join(lines)

    def _find_bib_entry(key):
        """Return the raw text of docs/references/*.bib's entry for ``key``, or None."""
        import re
        from pathlib import Path

        refs_dir = Path(env.conf["docs_dir"]) / "references"
        for bib_file in refs_dir.glob("*.bib"):
            text = bib_file.read_text()
            # Split on entry boundaries and find the one matching our key
            for candidate in re.split(r"\n(?=@)", text.strip()):
                m = re.match(r"@\w+\{([^,\s]+)", candidate)
                if m and m.group(1) == key:
                    return candidate.strip()
        return None

    @env.macro
    def bibtex_entry(key):
        """Look up ``key`` in docs/references/*.bib and render its raw entry in a ```bibtex fence.

        Falls back to a plain ``[@key]`` inline citation if no matching entry is found.
        """
        entry = _find_bib_entry(key)
        if not entry:
            return f"[@{key}]\n"
        return f"```bibtex\n{entry}\n```\n"

    def _paper_link(key):
        """DOI (preferred, as a doi.org URL) or URL field from ``key``'s own .bib entry, or
        None if the entry has neither -- so a paper-icon link never needs its own duplicated
        frontmatter field, just the same bib_key already used for the citation itself."""
        import re

        entry = _find_bib_entry(key)
        if not entry:
            return None
        doi = re.search(r'doi\s*=\s*[{"]([^}"]+)[}"]', entry, re.IGNORECASE)
        if doi:
            return f"https://doi.org/{doi.group(1)}"
        url = re.search(r'url\s*=\s*[{"]([^}"]+)[}"]', entry, re.IGNORECASE)
        return url.group(1) if url else None

    def _resolve_link(target, from_src_path):
        """Resolve ``target`` to a link valid from whatever page is currently being rendered.

        ``target`` is either an absolute URL (used as-is) or a path relative to ``docs_dir``
        root (e.g. ``"graphs/cgf_spec.md"``) -- resolved against ``from_src_path`` (typically
        ``env.page.file.src_path``, i.e. wherever this is actually being rendered right now, be
        that a standalone page or an accordion built by another page's macro), so the SAME
        frontmatter value produces a correct link regardless of which page embeds it.
        """
        if target.startswith(("http://", "https://")):
            return target
        from pathlib import PurePosixPath

        depth = len(PurePosixPath(from_src_path).parent.parts)
        return "../" * depth + target

    def _icon_row(meta, from_src_path, *, docs_page_target=None):
        """Badge-style icon links for a causal graph -- the same icon-row idea
        ``dataset_badges()`` already applies to dataset pages:

        - the paper (derived from ``bib_key``'s own doi/url field, if any)
        - its website (``meta["website"]``, if set -- an external homepage like causenet.org,
          or an internal page like the CGF spec; either way resolved via ``_resolve_link``)
        - the generated docs page for this graph under docs/graphs/graphs/*.md
          (``docs_page_target``, only passed by ``graph_accordions()`` -- omitted from a page's
          own icon row, since that page linking to itself would be pointless)
        """
        icons = []
        bib_key = meta.get("bib_key")
        if bib_key:
            paper_url = _paper_link(bib_key)
            if paper_url:
                icons.append(f"[:page_facing_up:]({paper_url})")
        website = meta.get("website")
        if website:
            icons.append(f"[:globe_with_meridians:]({_resolve_link(website, from_src_path)})")
        if docs_page_target:
            icons.append(f"[:link:]({_resolve_link(docs_page_target, from_src_path)})")
        return " ".join(icons)

    @env.macro
    def graph_page_icons():
        """Icon row for a docs/graphs/graphs/*.md page's own body -- see ``_icon_row``."""
        return _icon_row(env.page.meta, env.page.file.src_path)

    @env.macro
    def dataset_citation():
        bib_key = env.page.meta.get("bib_key")
        if not bib_key:
            return ""

        return f"## Citation\n\n{bibtex_entry(bib_key)}"

    @env.macro
    def task_datasets(task_id):
        """Build the dataset reference table for a task page from dataset frontmatter."""
        from pathlib import Path, PurePosixPath

        import yaml

        datasets_dir = Path(env.conf["docs_dir"]) / "datasets"

        # Relative path from the calling page's directory to docs/datasets/
        page_dir = PurePosixPath(env.page.file.src_path).parent
        depth = len(page_dir.parts)
        link_base = "../" * depth + "datasets"

        rows = []
        for md_file in sorted(datasets_dir.glob("*.md")):
            if md_file.name == "index.md":
                continue
            text = md_file.read_text()
            if not text.startswith("---"):
                continue
            end = text.index("---", 3)
            fm = yaml.safe_load(text[3:end]) or {}

            if task_id not in (fm.get("supported_tasks") or {}):
                continue

            title = fm.get("title", md_file.stem)
            doi = fm.get("doi")
            hf_page = fm.get("hf_page")
            bib_key = fm.get("bib_key")
            granularity = _GRANULARITY_LABELS.get(
                fm.get("granularity", _DEFAULT_GRANULARITY),
                fm.get("granularity"),
            )

            link = f"[{title}]({link_base}/{md_file.stem}.md)"
            ref = f"[@{bib_key}]" if bib_key else "—"
            badges = (
                " ".join(
                    p
                    for p in [
                        f"[:page_facing_up:](https://doi.org/{doi})" if doi else "",
                        f"[:hugging:]({hf_page})" if hf_page else "",
                    ]
                    if p
                )
                or "—"
            )
            rows.append(
                (
                    link,
                    fm.get("sentences", "—"),
                    fm.get("domain", "—"),
                    str(fm.get("year", "—")),
                    granularity,
                    ref,
                    badges,
                )
            )

        if not rows:
            return ""

        lines = [
            "| Corpus | Sentences | Domain | Year | Granularity | Reference | Links |\n",
            "|--------|----------:|--------|-----:|-------------|-----------|-------|\n",
        ]
        for corpus, sents, domain, year, granularity, ref, badges in rows:
            lines.append(f"| {corpus} | {sents} | {domain} | {year} | {granularity} | {ref} | {badges} |\n")
        return "".join(lines)

    @env.macro
    def graph_accordions():
        """Build one collapsible ``??? example`` block per file in docs/graphs/graphs/*.md, each
        showing that graph backend's description, usage snippet, and (if present) citation.

        Sourced entirely from each file's own frontmatter (``title``/``description``/
        ``snippet``/``bib_key``), read directly from disk -- the same pattern
        ``task_datasets`` already uses -- never from the file's rendered BODY. Each of those
        files is also its own standalone page using the same fields via ``{{ page.meta.* }}``;
        Jinja's ``page`` context inside a reused snippet resolves against the file being
        RENDERED (this one), not wherever the content originally came from, so anything meant
        to be reused across pages has to be a plain frontmatter value, not templated body text.
        Verified empirically: an embedded ``{{ page.meta... }}``/``{{ bibtex_entry(...) }}``
        call silently renders as literal, un-evaluated text when read this way, not an error.
        """
        import textwrap
        from pathlib import Path

        import yaml

        graphs_dir = Path(env.conf["docs_dir"]) / "graphs" / "graphs"
        blocks = []
        for slug in _GRAPH_ORDER:
            md_file = graphs_dir / f"{slug}.md"
            if not md_file.exists():
                continue
            text = md_file.read_text()
            if not text.startswith("---"):
                continue
            end = text.index("---", 3)
            fm = yaml.safe_load(text[3:end]) or {}
            title = fm.get("title", slug)
            description = (fm.get("description") or "").strip()
            snippet = (fm.get("snippet") or "").strip()
            bib_key = fm.get("bib_key")

            parts = []
            icons = _icon_row(fm, env.page.file.src_path, docs_page_target=f"graphs/graphs/{slug}.md")
            if icons:
                parts.append(icons)
            if description:
                parts.append(description)
            if snippet:
                parts.append(f"```python\n{snippet}\n```")
            if bib_key:
                parts.append("**Citation:**\n\n" + bibtex_entry(bib_key))

            block = f'??? example "{title}"\n\n' + textwrap.indent("\n\n".join(parts), "    ")
            blocks.append(block)
        return "\n\n".join(blocks)
