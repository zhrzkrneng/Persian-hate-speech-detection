# The English manuscript (ACL format)

    python -m scripts.build_english_paper          # main.tex.in -> main.tex
    python -m scripts.build_english_paper --check  # report drift, write nothing

`main.tex.in` is the source. `main.tex` is **generated** — edit the template,
never the output.

## No number here is typed

Every statistic is injected from `runs/paper_numbers.json`, which is computed
from the committed per-seed records. A placeholder the renderer has no value
for is an error, not an empty string, and `tests/test_english_paper.py` fails
if a decimal is typed into the template at all.

This is not pedantry. The one number that *was* typed in the Persian
manuscript — "five seeds" in its English abstract — stayed wrong for weeks
after the sweep to fifteen and survived two rounds of review.

So after new runs land:

    python -m scripts.paper_numbers
    python -m scripts.build_english_paper

## Building the PDF

There is no TeX toolchain in the cloud container, and installing one to
typeset a paper Overleaf typesets for free is a poor trade. Build it there:

1. Open the official **ACL template** on Overleaf (ACL publishes a current one
   each cycle; use the cycle you are submitting to, since `acl.sty` changes).
2. Replace its `.tex` with `main.tex` from this directory and upload
   `references.bib` beside it.
3. Keep the template's own `acl.sty`, `acl_natbib.bst` and any font files.

`\usepackage[review]{acl}` is set, which is what an anonymous ARR submission
wants: it adds line numbers and keeps the author block anonymous. Switch to
`\usepackage{acl}` for a camera-ready.

## Page limit

ACL long papers allow 8 pages of content. **Limitations and the Ethics
Statement do not count**, nor do references or the appendix. The Limitations
section is mandatory at ACL venues — it is `\section*`, unnumbered, and it is
already in the template.

If the body runs over, move detail to the appendix rather than cutting the
statistics: the per-seed records, the multiplicity table and the equivalence
test are what make the claims checkable.

## What the tests check

`tests/test_english_paper.py` stands in for a compiler, which is not available
here. It checks that braces balance, that math mode closes, that every
environment is closed by its own name, that each table row has as many cells
as its preamble has columns, that every `\cite` key exists in
`references.bib`, that the Limitations section is present, and that the
generated file matches the template and the records.

It does **not** prove the document compiles. Compile it on Overleaf before
submitting and read the log.

## Anonymity

For an ARR submission the paper must be anonymous. The author block is already
`Anonymous ACL submission`, but check the rest before uploading: the
Reproducibility appendix, the Ethics Statement and any repository URL can
deanonymise a paper. ARR also has an anonymity-period policy covering
preprints; read the current one rather than assuming.
