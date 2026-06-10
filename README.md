<p align="center">
   <img width=200px src="docs/assets/icon.png"/>
</p>

# Installation

```bash
pip install git+https://github.com/TheMrSheldon/causality-toolkit.git
```

# Building the Documentation

Install the docs dependencies and serve locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

To build the static site:

```bash
mkdocs build
```

The output is written to the `site/` directory.