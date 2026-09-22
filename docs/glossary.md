# Glossary

## Relation

A **relation** is a set of ordered pairs over a set of entities, events, or concepts *E* --
formally, a subset of *E* &times; *E*, matching the standard set-theoretic / relational-algebra
sense of the word. causalatee uses "relation" this way to name a *type* of connection: the
**causal relation**, the **countercausal relation** (see
[`causalatee.data.constants.Relation`][causalatee.data.constants.Relation]).

## Relationship

A **relationship** is one *element* of a relation -- a single ordered pair *(x, y)* that belongs
to it, not the relation itself. "*x* causes *y*" expresses a causal **relationship** between *x*
and *y*: i.e., *(x, y)* is an element of the causal relation.

This distinction is deliberate throughout causalatee's API: `Relation` (the enum) names relation
*types*, while a `relationship` field on a per-pair record --
[`IdentifiedRelation.relationship`][causalatee.models.IdentifiedRelation] --
names *one pair's* classification into one of those types. Read "the causal relation between
smoking and cancer" as the general, type-level claim, and "there is a causal relationship between
*this* smoking event and *this* cancer diagnosis" as one concrete instance of it.

## Entity

One element of *E* -- an entity, event, or concept that can participate as the cause or the
effect side of a relationship. Entities are the spans marked in a source text (e.g. via
[`insert_entity_markers`][causalatee.data.utils.insert_entity_markers]) that a relationship's
`first`/`second` (or `e1`/`e2`) fields reference by id.
