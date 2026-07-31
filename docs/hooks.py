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


def define_env(env):
    @env.macro
    def dataset_pills():
        """Small colored ``<span class="pill pill-...">`` tags summarising
        at-a-glance dataset properties (currently just granularity — see
        ``_GRANULARITY_LABELS`` — but built to take more fields later)."""
        meta = env.page.meta
        granularity_key = meta.get("granularity", _DEFAULT_GRANULARITY)
        label = _GRANULARITY_LABELS.get(granularity_key, granularity_key)
        return f'<span class="pill pill-{granularity_key}">{label}</span>\n\n'

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
            lines.append(
                f'<iframe src="{url}" frameborder="0" width="100%" height="560px"></iframe>\n\n'
            )
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
                f'<div class="tira-leaderboard" data-tira-url="{url}">'
                "<p><em>Loading leaderboard…</em></p></div>\n\n"
            )
        return "".join(lines)

    @env.macro
    def bibtex_entry(key):
        """Look up ``key`` in docs/references/*.bib and render its raw entry in a ```bibtex fence.

        Falls back to a plain ``[@key]`` inline citation if no matching entry is found.
        """
        import re
        from pathlib import Path

        refs_dir = Path(env.conf["docs_dir"]) / "references"
        entry = None
        for bib_file in refs_dir.glob("*.bib"):
            text = bib_file.read_text()
            # Split on entry boundaries and find the one matching our key
            for candidate in re.split(r"\n(?=@)", text.strip()):
                m = re.match(r"@\w+\{([^,\s]+)", candidate)
                if m and m.group(1) == key:
                    entry = candidate.strip()
                    break
            if entry:
                break

        if not entry:
            return f"[@{key}]\n"

        return f"```bibtex\n{entry}\n```\n"

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
            badges = " ".join(
                p for p in [
                    f"[:page_facing_up:](https://doi.org/{doi})" if doi else "",
                    f"[:hugging:]({hf_page})" if hf_page else "",
                ]
                if p
            ) or "—"
            rows.append((
                link,
                fm.get("sentences", "—"),
                fm.get("domain", "—"),
                str(fm.get("year", "—")),
                granularity,
                ref,
                badges,
            ))

        if not rows:
            return ""

        lines = [
            "| Corpus | Sentences | Domain | Year | Granularity | Reference | Links |\n",
            "|--------|----------:|--------|-----:|-------------|-----------|-------|\n",
        ]
        for corpus, sents, domain, year, granularity, ref, badges in rows:
            lines.append(f"| {corpus} | {sents} | {domain} | {year} | {granularity} | {ref} | {badges} |\n")
        return "".join(lines)
