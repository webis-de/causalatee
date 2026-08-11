# causalatee.models

Structural (`typing.Protocol`) interfaces for causality models — one per
[task](../tasks/causality_detection.md), plus
[`PairwiseIdentification`][causalatee.models.PairwiseIdentification] (what nearly every
trained relation classifier actually implements) and
[`Extraction`][causalatee.models.Extraction] (the end-to-end composition of all three
subtasks). A class satisfies a protocol purely by having a matching `__call__`
signature — no inheritance required. See each task page's "Models" section for which
protocol applies there.

::: causalatee.models
    options:
      members_order: source
      show_if_no_docstring: false
      filters: ["!^_"]
