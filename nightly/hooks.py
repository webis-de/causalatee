"""mkdocs-macros hooks — generate repetitive dataset page sections from YAML frontmatter."""

import shutil
from pathlib import Path

import yaml


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
    "causality-extraction": "Causality Extraction",
}

_TASK_LINKS = {
    "causality-detection": "../tasks/causality_detection.md",
    "causal-candidate-extraction": "../tasks/causal_event_candidate_detection.md",
    "causality-identification": "../tasks/causality_identification.md",
    "causality-extraction": "../tasks/causality_extraction.md",
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


def _note_icon(meta):
    """Info-icon tooltip for a dataset page's frontmatter ``note`` field (default unset/None).

    Use sparingly -- only for something that makes a dataset genuinely stand out, not a routine
    fact already covered by the pill/badge system (task support, polarity/strength, DOI, ...).
    Renders as a native browser tooltip (no JS) via ``<abbr title="...">``, so ``note`` must be
    plain text -- HTML attribute values can't contain markdown links or other markup. Shown only
    in the dataset *tables* (``all_datasets()``), not on a dataset's own detail page -- the fact
    itself belongs in that page's body prose instead, where it has room to be written out in full.
    """
    note = meta.get("note")
    if not note:
        return ""
    escaped = note.replace("&", "&amp;").replace('"', "&quot;")
    return f'<abbr title="{escaped}">:material-information-outline:</abbr>'


def _find_bib_entry(key, docs_dir):
    """Return the raw text of docs/references/*.bib's entry for ``key``, or None."""
    import re

    refs_dir = Path(docs_dir) / "references"
    for bib_file in refs_dir.glob("*.bib"):
        text = bib_file.read_text()
        # Split on entry boundaries and find the one matching our key
        for candidate in re.split(r"\n(?=@)", text.strip()):
            m = re.match(r"@\w+\{([^,\s]+)", candidate)
            if m and m.group(1) == key:
                return candidate.strip()
    return None


def _paper_link(key, docs_dir):
    """DOI (preferred, as a doi.org URL) or URL field from ``key``'s own .bib entry, or
    None if the entry has neither -- so a paper-icon link never needs its own duplicated
    frontmatter field, just the same bib_key already used for the citation itself."""
    import re

    entry = _find_bib_entry(key, docs_dir)
    if not entry:
        return None
    doi = re.search(r'doi\s*=\s*[{"]([^}"]+)[}"]', entry, re.IGNORECASE)
    if doi:
        return f"https://doi.org/{doi.group(1)}"
    url = re.search(r'url\s*=\s*[{"]([^}"]+)[}"]', entry, re.IGNORECASE)
    return url.group(1) if url else None


def _badges_for(fm, docs_dir):
    """Shared paper/hf/repo icon list for a dataset or model's frontmatter -- used by
    ``dataset_badges()``, ``model_badges()``, ``all_datasets()``, and ``all_models()``, so all
    four stay in sync. The paper icon is always derived from the ``bib_key``'s own .bib entry
    (doi, then url) -- there's no separate ``doi`` frontmatter field to keep in sync with it."""
    hf_page = fm.get("hf_page")
    repo = fm.get("repo")
    bib_key = fm.get("bib_key")
    parts = []
    paper_url = _paper_link(bib_key, docs_dir) if bib_key else None
    if paper_url:
        parts.append(f"[:page_facing_up:]({paper_url})")
    if hf_page:
        parts.append(f"[:hugging:]({hf_page})")
    if repo:
        parts.append(f"[:material-git:]({repo})")
    return parts


def _iter_dataset_frontmatter(docs_dir):
    """Yield ``(md_file, frontmatter_dict)`` for every docs/datasets/*.md page except index.md --
    the shared source both ``task_datasets()`` and ``all_datasets()`` build their tables from."""
    datasets_dir = Path(docs_dir) / "datasets"
    for md_file in sorted(datasets_dir.glob("*.md")):
        if md_file.name == "index.md":
            continue
        text = md_file.read_text()
        if not text.startswith("---"):
            continue
        end = text.index("---", 3)
        fm = yaml.safe_load(text[3:end]) or {}
        yield md_file, fm


def _iter_model_frontmatter(docs_dir):
    """Yield ``(md_file, frontmatter_dict)`` for every docs/models/*.md page except index.md --
    the source ``all_models()`` builds its table from, mirroring ``_iter_dataset_frontmatter``."""
    models_dir = Path(docs_dir) / "models"
    for md_file in sorted(models_dir.glob("*.md")):
        if md_file.name == "index.md":
            continue
        text = md_file.read_text()
        if not text.startswith("---"):
            continue
        end = text.index("---", 3)
        fm = yaml.safe_load(text[3:end]) or {}
        yield md_file, fm


_TASK_CODES = {
    "causality-detection": "D",
    "causal-candidate-extraction": "E",
    "causality-identification": "I",
}


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
        parts = _badges_for(env.page.meta, env.conf["docs_dir"])
        return (" ".join(parts) + "\n\n") if parts else ""

    @env.macro
    def model_badges():
        """Icon row for a model's own detail page -- the same idea as ``dataset_badges()``."""
        parts = _badges_for(env.page.meta, env.conf["docs_dir"])
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

    @env.macro
    def bibtex_entry(key):
        """Look up ``key`` in docs/references/*.bib and render its raw entry in a ```bibtex fence.

        Falls back to a plain ``[@key]`` inline citation if no matching entry is found.
        """
        entry = _find_bib_entry(key, env.conf["docs_dir"])
        if not entry:
            return f"[@{key}]\n"
        return f"```bibtex\n{entry}\n```\n"

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
            paper_url = _paper_link(bib_key, env.conf["docs_dir"])
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
    def all_datasets(filter_task=None):
        """Build a dataset reference table from every docs/datasets/*.md page's frontmatter.

        Unfiltered (docs/datasets/index.md): every dataset, with a Tasks column (D/E/I) showing
        which tasks each one supports. Filtered to one ``filter_task`` (a task page, e.g.
        ``{{ all_datasets(filter_task="causality-detection") }}``): only datasets supporting that
        task, with Granularity/Reference columns instead of Tasks -- redundant once every row
        already supports the same one task.
        """
        from pathlib import PurePosixPath

        # Relative path from the calling page's directory to docs/datasets/
        page_dir = PurePosixPath(env.page.file.src_path).parent
        depth = len(page_dir.parts)
        link_base = "../" * depth + "datasets"

        rows = []
        for md_file, fm in _iter_dataset_frontmatter(env.conf["docs_dir"]):
            tasks = fm.get("supported_tasks") or {}
            if filter_task is not None and filter_task not in tasks:
                continue

            title = fm.get("title", md_file.stem)
            link = f"[{title}]({link_base}/{md_file.stem}.md)"
            note_icon = _note_icon(fm)
            if note_icon:
                link += f" {note_icon}"
            rows.append(
                {
                    "link": link,
                    "domain": fm.get("domain", "—"),
                    "year": str(fm.get("year", "—")),
                    "sentences": fm.get("sentences", "—"),
                    "granularity": _GRANULARITY_LABELS.get(
                        fm.get("granularity", _DEFAULT_GRANULARITY), fm.get("granularity")
                    ),
                    "ref": f"[@{fm['bib_key']}]" if fm.get("bib_key") else "—",
                    # causality-extraction is intentionally excluded here, not defaulted to "?":
                    # it's a derived capability (any dataset with causality-identification data
                    # already supports the composed end-to-end task via compose_extraction), not a
                    # separately stored task config, so it would just duplicate the "I" it implies.
                    "task_codes": "".join(_TASK_CODES[t] for t in tasks if t in _TASK_CODES) or "—",
                    "badges": " ".join(_badges_for(fm, env.conf["docs_dir"])) or "—",
                }
            )

        if not rows:
            return ""

        if filter_task is not None:
            lines = [
                "| Corpus | Sentences | Domain | Year | Granularity | Reference | Links |\n",
                "|--------|----------:|--------|-----:|-------------|-----------|-------|\n",
            ]
            for r in rows:
                lines.append(
                    f"| {r['link']} | {r['sentences']} | {r['domain']} | {r['year']} | "
                    f"{r['granularity']} | {r['ref']} | {r['badges']} |\n"
                )
        else:
            lines = [
                "**Task codes**: D = Detection · E = Extraction · I = Identification\n\n",
                "| Dataset | Domain | Year | Sentences | Tasks | Links |\n",
                "|---------|--------|------|----------:|-------|-------|\n",
            ]
            for r in rows:
                lines.append(
                    f"| {r['link']} | {r['domain']} | {r['year']} | {r['sentences']} | "
                    f"{r['task_codes']} | {r['badges']} |\n"
                )
        return "".join(lines)

    @env.macro
    def all_models():
        """Build a model reference table from every docs/models/*.md page's frontmatter,
        mirroring ``all_datasets()``'s unfiltered table (task-code legend, Links badges)."""
        from pathlib import PurePosixPath

        page_dir = PurePosixPath(env.page.file.src_path).parent
        depth = len(page_dir.parts)
        link_base = "../" * depth + "models"

        rows = []
        for md_file, fm in _iter_model_frontmatter(env.conf["docs_dir"]):
            title = fm.get("title", md_file.stem)
            tasks = fm.get("supported_tasks") or {}
            rows.append(
                {
                    "link": f"[{title}]({link_base}/{md_file.stem}.md)",
                    "type": fm.get("type", "—"),
                    "task_codes": "".join(_TASK_CODES[t] for t in tasks if t in _TASK_CODES) or "—",
                    "ref": f"[@{fm['bib_key']}]" if fm.get("bib_key") else "—",
                    "badges": " ".join(_badges_for(fm, env.conf["docs_dir"])) or "—",
                }
            )

        if not rows:
            return ""

        lines = [
            "**Task codes**: D = Detection · E = Extraction · I = Identification\n\n",
            "| Model | Type | Tasks | Reference | Links |\n",
            "|-------|------|-------|-----------|-------|\n",
        ]
        for r in rows:
            lines.append(f"| {r['link']} | {r['type']} | {r['task_codes']} | {r['ref']} | {r['badges']} |\n")
        return "".join(lines)

    @env.macro
    def graph_cards():
        """Build a ``<div class="grid cards">`` navigation tile per file in
        docs/graphs/graphs/*.md -- the same grid-cards convention docs/examples/index.md already
        uses. Each card only teases (icon, title, one-line description) and links out to that
        graph's own page; it doesn't duplicate the icon row/snippet/citation, since that page
        already renders all of it from the exact same frontmatter -- two copies of the same
        content would just be two places to keep in sync.

        Sourced entirely from each file's own frontmatter (``title``/``description``/``icon``),
        read directly from disk -- the same pattern ``_iter_dataset_frontmatter`` already uses --
        never from the file's rendered BODY.
        """
        from pathlib import Path

        import yaml

        graphs_dir = Path(env.conf["docs_dir"]) / "graphs" / "graphs"
        cards = []
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
            icon = fm.get("icon", "material-graph-outline")
            link = _resolve_link(f"graphs/graphs/{slug}.md", env.page.file.src_path)

            cards.append(
                f"-   :{icon}:{{ .lg .middle }} **{title}**\n\n"
                "    ---\n\n"
                f"    {description}\n\n"
                f"    [View details →]({link})"
            )

        if not cards:
            return ""
        return '<div class="grid cards" markdown>\n\n' + "\n\n".join(cards) + "\n\n</div>\n"
