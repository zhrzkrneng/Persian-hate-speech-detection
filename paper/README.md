# The manuscript

> **The built `.docx` files are not committed while the paper is under
> review.** `paper/*.docx` is gitignored for that reason. Everything needed to
> rebuild them is here, so `bash paper/build.sh --all` reproduces all three
> from the records. Restore them to the repository once the paper is accepted,
> or link the published version instead.


    bash paper/build.sh                  # -> paper/maghale.docx
    bash paper/build.sh gozaresh.docx    # -> paper/gozaresh.docx (the report)
    bash paper/build.sh --all            # both

Persian, two columns, built against the journal's own template
(«نشریه پژوهش‌های نظری و کاربردی هوش ماشینی»): B Lotus for Persian and Times
New Roman one point smaller for Latin, at the sizes its table of styles gives;
numbered sections from the introduction; table captions above, figure captions
below; IEEE numeric references.

**No number in this document is typed.** `scripts/paper_numbers.py` collects
every mean, spread, paired test, interaction, recall and chance level from the
committed per-seed records into `runs/paper_numbers.json`, and the generator
reads only that. The figure is drawn from the same file, which it was not at
first: it was redrawn while two encoders were being re-run and came back with
two points beside tables that still had four.

So a rebuild after new runs land is:

    python -m scripts.paper_numbers
    bash paper/build.sh

`--fallback-ref <rev>` fills a cell missing from the working tree from a git
revision and lists it under `provisional`; the generator then prints a notice
at the top of the document. It exists so a draft during a rebuild cannot
quietly mix pipelines.

## Three things the docx library cannot do

`build.sh` inserts `<w:bidi/>` into each `<w:sectPr>` after the build. Column
order is a section property, not a paragraph one, and without it the first
column lands on the left — a Persian reader then meets the sections out of
order, which is how the first draft read.

**Call `build.sh`, never a generator directly.** `node report_build.js`
produces a gozaresh.docx that opens, validates and reads in the wrong
direction, because the bidi step lives here and not in the generator. That is
how the report lost its section bidi once already, so the step now fails loudly
if no `<w:sectPr>` took it.

It also rezips with `-D`. A docx that carries directory entries opens in Word
but has failed to open through cloud services — which is where a journal
portal and a co-author's browser both read it — and every build before this
one carried five.

Numeric table cells are laid out left-to-right. In a right-to-left paragraph a
leading sign is reordered to the far side: `-1.53` renders as `1.53-` and
`n.s.` as `.n.s`. Inline statistics in running Persian carry U+200E for the
same reason.

**A long statistic goes in a table cell, not in a sentence.** `0.0019` in
running text came out as `0.001` ending one line and `9.` starting the next.
U+2060 word joiners between the digits were tried; LibreOffice broke at them
anyway, so the only renderer available here could not confirm the fix. Keep
four-decimal numbers out of running Persian text and put them in a column.

**Table rows that cross the column break need `cantSplit: true`.** Without it
a two-line label is torn in half across the break -- `برهم‌کنش: B/32 −`
ending one column and `jina` opening the next. Set it on every row of a table
long enough to cross.

## Verify a build

    python scripts/office/validate.py paper/maghale.docx   # from the docx skill
    soffice --headless --convert-to pdf paper/maghale.docx
    pdftoppm -jpeg -r 90 paper/maghale.pdf page            # then look at them

Looking is not optional. The first two faults above produced a file that
opened, passed schema validation, and was wrong; the third is the one the
validator does report, as a note rather than a failure, so read its notes too.
