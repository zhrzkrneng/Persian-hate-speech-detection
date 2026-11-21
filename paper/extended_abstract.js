// The extended English abstract (چکیده مبسوط) Iranian journals ask for
// alongside a Persian manuscript. Built as its own single-column, LTR
// document so it can be submitted as a separate file, and generated from
// runs/paper_numbers.json like everything else: the ordinary abstract carried
// "five seeds" for weeks after the sweep because it was typed, and a second
// hand-maintained summary of the same results would go the same way.
const B = require("./build.js");
const { Document, Packer, Paragraph, TextRun, AlignmentType, SIZE, en, fmt,
        signed, N, fs } = B;

const SHORT = {
  "CLIP ViT-L/14": "CLIP ViT-L/14",
  "CLIP ViT-B/32": "CLIP ViT-B/32",
  "jina-clip-v2": "jina-clip-v2",
  CLIPfa: "CLIPfa",
};
const au = (e, k) => N.cells[e][k].test.auroc.mean;
const ben = (e) => N.fusion_benefit[e];
const arch = (e) => N.architecture[e];
const ret = (e, l) => N.retrieval[e]?.[l]?.t2i_r1;
const SEEDS = ben("CLIP ViT-L/14").n;
const MU = N.multiplicity;
const inter = N.interaction["CLIP ViT-L/14 - CLIP ViT-B/32"];
const ml = N.multilingual;

const P = (text, opts = {}) =>
  new Paragraph({
    bidirectional: false,
    alignment: opts.align || AlignmentType.JUSTIFIED,
    spacing: { after: opts.after === undefined ? 120 : opts.after },
    children: [new TextRun(en(text, { size: SIZE.body, bold: !!opts.bold }))],
  });

const H = (text) =>
  new Paragraph({
    bidirectional: false,
    alignment: AlignmentType.LEFT,
    spacing: { before: 180, after: 80 },
    children: [new TextRun(en(text, { size: SIZE.body, bold: true }))],
  });

const body = [];

body.push(
  new Paragraph({
    bidirectional: false,
    alignment: AlignmentType.CENTER,
    spacing: { after: 160 },
    children: [
      new TextRun(
        en("Extended Abstract", { size: SIZE.h1 || SIZE.body, bold: true }),
      ),
    ],
  }),
);

body.push(H("Introduction"));
body.push(
  P(
    "Hateful memes carry their meaning across two modalities at once: an " +
      "image that is innocuous alone and a caption that is innocuous alone " +
      "can be hateful together. The architectures that define this task " +
      "therefore combine the two feature vectors explicitly, most commonly " +
      "by an element-wise product of a projected image vector and a " +
      "projected text vector, a fusion its authors call align. Hate-CLIPper " +
      "reports that this fusion gives no advantage over plain concatenation " +
      "for Tamil, a language its encoder never saw, and attributes the " +
      "failure to misaligned feature spaces. Persian has the same problem " +
      "and no public hateful-meme dataset at all. The attribution is widely " +
      "repeated and, as far as we can establish, had never been tested: it " +
      "names a cause without measuring it. This paper tests it.",
  ),
);

body.push(H("Methodology"));
body.push(
  P(
    `Four frozen encoders are compared on the Hateful Memes Challenge: ` +
      `${SHORT["CLIP ViT-L/14"]} and ${SHORT["CLIP ViT-B/32"]}, which are ` +
      `English-centred and differ in capacity; jina-clip-v2, which is ` +
      `multilingual; and CLIPfa, which is Persian-only. Each encoder is run ` +
      `under two fusions, concatenation and align, with every one of the ` +
      `nine hyperparameters of the reference architecture held fixed, and ` +
      `each cell is trained over ${SEEDS} random seeds. Because the encoders ` +
      `are frozen, their features are encoded once and cached, which is what ` +
      `makes a sweep of this size affordable without dedicated hardware.`,
  ),
);
body.push(
  P(
    "The study's second axis is a measurement no training produces. Each " +
      "encoder's cross-modal alignment is obtained by zero-shot retrieval on " +
      "Crossmodal-3600, which asks whether the encoder places a caption near " +
      "its own image with no classifier and no labels involved. On the " +
      "1,000-image slice used here chance is 0.10 percent, and every recall " +
      "is read against it. This gives alignment and classification accuracy " +
      "as independent quantities rather than two readings of one pipeline.",
  ),
);
body.push(
  P(
    "All comparisons are paired over shared seeds and reported with a " +
      "paired t-test, an effect size and a confidence interval. Claims that " +
      "two quantities do not differ are made with two one-sided tests " +
      "against a margin fixed before the numbers were seen, because a " +
      "non-significant p-value is not evidence of equivalence. The " +
      `${MU.family_size} tests that carry the results section are corrected ` +
      "for multiplicity by the Holm-Bonferroni procedure.",
  ),
);

body.push(H("Results and Discussion"));
body.push(
  P(
    `The fusion benefit changes sign with the encoder. It is ` +
      `${signed(ben("CLIP ViT-L/14").mean)} AUROC on ${SHORT["CLIP ViT-L/14"]} ` +
      `and ${signed(ben("CLIP ViT-B/32").mean)} on ${SHORT["CLIP ViT-B/32"]}, ` +
      `both positive; ${signed(ben("jina-clip-v2").mean)} on jina-clip-v2 and ` +
      `${signed(ben("CLIPfa").mean)} on CLIPfa, both negative. All four are ` +
      `significant after correction and all four are large relative to seed ` +
      `noise. The phenomenon the reference paper reports is therefore real ` +
      `and is not confined to one unseen language.`,
  ),
);
body.push(
  P(
    `The decisive control separates alignment from capacity. ` +
      `${SHORT["CLIP ViT-L/14"]} and ${SHORT["CLIP ViT-B/32"]} differ by ` +
      `${fmt(au("CLIP ViT-L/14", "hateclipper/concat") - au("CLIP ViT-B/32", "hateclipper/concat"))} ` +
      `AUROC under concatenation, a large gap in this table, while their ` +
      `measured alignment differs by about one point of recall. Their fusion ` +
      `benefits are equivalent: the paired difference is ` +
      `${signed(inter.mean)} with a 95% interval of [${signed(inter.ci_low)}, ` +
      `${signed(inter.ci_high)}], and the equivalence test rejects an effect ` +
      `of a point or more. Capacity is not what decides.`,
  ),
);
body.push(
  P(
    "The explanation, however, does not survive its own test. If the " +
      "benefit were a property of alignment, it should be predictable from " +
      "the alignment of each encoder-language pair. A head trained on " +
      "English memes reads 300 parallel memes zero-shot in five languages, " +
      `giving twenty cells. The correlation is ` +
      `r = ${fmt(ml.overall.pearson_r, 3)} (p = ${fmt(ml.overall.pearson_p, 3)}), ` +
      `that is, none. Neither dropping the cells near chance, nor correcting ` +
      `for measurement error, which bounds the true correlation at ` +
      `|r| <= ${fmt(ml.attenuation_bound, 3)}, nor fitting each language ` +
      `separately recovers a relationship. The same encoder dependence ` +
      `appears on a second axis: the three additions of MemeCLIP help on ` +
      `${SHORT["CLIP ViT-L/14"]} (${signed(arch("CLIP ViT-L/14").mean)}) and ` +
      `hurt on jina-clip-v2 (${signed(arch("jina-clip-v2").mean)}), both ` +
      `significantly, so the sign flips with the encoder there too.`,
  ),
);

body.push(H("Conclusion"));
body.push(
  P(
    "What this work leaves is a confirmed phenomenon and a refuted " +
      "explanation. Whether cross-modal fusion is worth its cost depends on " +
      "the encoder, strongly enough to change sign, and the measured " +
      "alignment of the feature spaces is not what it depends on. The " +
      "practical consequence for a low-resource language is that an " +
      "architecture cannot be chosen from a result obtained on another " +
      "encoder, and that the fusion must be ablated per encoder rather than " +
      "assumed. No public Persian hateful-meme dataset exists, this study " +
      "had no annotation capacity to build one, and no Persian number is " +
      "reported; what is offered instead is the measurement a Persian " +
      "dataset would be evaluated against. Every number in this paper is " +
      "generated from committed per-seed records, and the code and " +
      "configurations are public.",
  ),
);
body.push(
  P(
    "Keywords: hateful memes, multimodal learning, frozen encoders, " +
      "cross-modal alignment, low-resource languages, Persian NLP",
    { bold: true, after: 0 },
  ),
);

async function main() {
  // build.js reads argv[2] as the numbers file and argv[3] as its output, so
  // this takes the same shape: node extended_abstract.js <numbers> <out>.
  const out = process.argv[3] || "extended-abstract.docx";
  const doc = new Document({
    sections: [{ properties: {}, children: body }],
  });
  const buf = await Packer.toBuffer(doc);
  fs.writeFileSync(out, buf);
  console.log(`wrote ${out} (${(buf.length / 1024).toFixed(0)} KB)`);
}

module.exports = { body };
if (require.main === module) main();
