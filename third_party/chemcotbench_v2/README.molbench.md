# Vendored ChemCoTBench-V2 evaluator

This directory contains the evaluator, formal-CoT parsers/verifiers, and
minimal chemistry utilities from `fresnellll/ChemCoTBench-V2` at commit
`dcd35470de4096a1b10ee9ed6f072bcee983a9cc`.

The upstream code is MIT licensed; the original `LICENSE` and `CITATION.cff`
are included. MolBench calls these modules through a small adapter and keeps
the upstream metric definitions unchanged.

MolBench removes the eager `runner`/`sampler` imports from
`evaluation.core.__init__`. Those imports pull in the OpenAI client even for
offline parsing; parser, verifier, and metric implementations are unchanged.
