---
title: SemEval-2023 Task 8 (RedHOT)
domain: Social media / Medical
year: 2023
sentences: "—"
supported_tasks: {}
---

<!-- TODO(user): add the RedHOT/SemEval-2023 Task 8 bibtex entry to
     docs/references/datasets.bib (Wadhwa et al., arXiv:2210.06331,
     "Redhot: A Corpus of Annotated Medical Questions, Experiences, and
     Claims on Social Media"), then set bib_key: in the frontmatter above
     and add {{ dataset_citation() }} to the bottom of this page. -->

# {{ page.meta.title }}

{{ dataset_badges() }}
{{ dataset_pills() }}

SemEval-2023 Task 8 asks systems to identify causal medical claims in Reddit posts and extract related PICO (Population/Intervention/Comparator/Outcome) spans, built on the RedHOT (Reddit Health Online Talk) corpus.

!!! warning "Blocked — no full, directly-fetchable labeled dataset"
    The official train/test split is distributed only via a CodaLab competition, gated behind shared-task
    account registration. The full underlying RedHOT corpus requires a self-attested Institutional Review
    Board (IRB) approval checkbox plus running a script that fetches the real Reddit comment text live via
    the Reddit API at download time (its own GitHub repo, `github.com/sominw/redhot`, currently 404s). A
    small (~178-row) public sample exists on Google Drive in the real annotation schema, but Drive folder
    listings aren't fetchable via a plain scripted request the way this project's other sources are, and the
    sample is far too small to be a usable dataset regardless. Not yet attempted — see
    `resources/datasets/SemEval2023T8/conversion_script.py` for the full trail of what was checked.

{{ dataset_overview() }}
