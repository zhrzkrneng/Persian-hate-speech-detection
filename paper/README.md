# The manuscript

    bash paper/build.sh            # -> paper/maghale.docx

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

## Two things the docx library cannot do

`build.sh` inserts `<w:bidi/>` into each `<w:sectPr>` after the build. Column
order is a section property, not a paragraph one, and without it the first
column lands on the left — a Persian reader then meets the sections out of
order, which is how the first draft read.

Numeric table cells are laid out left-to-right. In a right-to-left paragraph a
leading sign is reordered to the far side: `-1.53` renders as `1.53-` and
`n.s.` as `.n.s`. Inline statistics in running Persian carry U+200E for the
same reason.

## Verify a build

    python scripts/office/validate.py paper/maghale.docx   # from the docx skill
    soffice --headless --convert-to pdf paper/maghale.docx
    pdftoppm -jpeg -r 90 paper/maghale.pdf page            # then look at them

Looking is not optional. Both faults above produced a file that opened, passed
schema validation, and was wrong.
