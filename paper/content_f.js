// Material a referee would ask for and the draft did not have: what the head
// actually is, why AUROC, the raw per-seed differences, why retrieval is an
// honest proxy for alignment, and the ethical frame.
const B = require("./build.js");
const { P, H, caption, cell, table, T, TL, fmt, signed, ltr, N,
        TableRow, Paragraph, TextRun, AlignmentType, fa, en, SIZE } = B;

const ENC = ["CLIP ViT-L/14", "CLIP ViT-B/32", "jina-clip-v2", "CLIPfa"];
const ben = (e) => N.fusion_benefit[e];

// ---- goes into §2, after the existing related-work paragraphs ----
const related = [];
related.push(P([fa(
  "آنچه این سه خط پژوهشی در آن مشترک‌اند، و آنچه این مقاله به آن می‌پردازد، " +
  "این است: هر سه رمزگذار را ثابت فرض می‌کنند و درباره‌ی سازوکارِ ترکیب " +
  "بحث می‌کنند. اگر خاصیتِ رمزگذار تعیین کند که ترکیب چقدر می‌ارزد، آنگاه " +
  "نتیجه‌ای که روی یک رمزگذار به دست آمده درباره‌ی رمزگذار دیگر چیزی " +
  "نمی‌گوید، و این موضوعی است که با یک رمزگذار قابل آزمودن نیست."),
]));

// ---- goes into §3 as a new subsection ----
const method = [];
method.push(H("۳-۴ سرِ رده‌بند و بودجه‌ی آموزش", 2));
method.push(P([fa(
  "پس از ترکیب، یک لایه‌ی پنهان با گذرِ غیرخطی و حذف تصادفی قرار دارد و " +
  "سپس لایه‌ی خروجیِ دوکلاسه. معماری "), en("MemeCLIP"), fa(
  " همین مسیر را با سه افزوده دارد: وفق‌دهنده‌ی پس‌ماندی روی هر مُد پیش از " +
  "ترکیب، نرمال‌سازی "), en("L2"), fa(
  " بر هر بردار، و رده‌بندی که شباهتِ کسینوسی را با یک ضریبِ مقیاس به " +
  "لوجیت تبدیل می‌کند."),
], { indent: { firstLine: 0 } }));

method.push(P([fa(
  "بودجه‌ی آموزش برای همه‌ی سلول‌ها یکسان است: بیست دوره، بهینه‌سازِ " +
  "تطبیقی با واپاشیِ وزن، بریدنِ گرادیان، و دسته‌های با اندازه‌ی ثابت. " +
  "هیچ جست‌وجوی فراپارامتری به‌ازای رمزگذار انجام نشده است. این انتخاب " +
  "عمدی است: جست‌وجوی جداگانه برای هر رمزگذار، تفاوتِ میان آن‌ها را با " +
  "تفاوت در کوششِ تنظیم می‌آمیخت. بهای آن، سطحِ مطلقِ پایین‌تر است، که در " +
  "بخش ۷ آمده."),
]));

// ---- goes into §4 ----
const metrics = [];
metrics.push(H("۴-۱ معیار ارزیابی", 2));
metrics.push(P([en("AUROC"), fa(
  " معیار اصلی است، همان‌که در پیکره‌ی اصلی و در هر دو معماریِ مرجع گزارش " +
  "می‌شود. دلیلش این است که به آستانه وابسته نیست: دو مدل با توانِ " +
  "رتبه‌بندیِ یکسان ولی کالیبراسیونِ متفاوت، "), en("AUROC"), fa(
  " یکسان می‌گیرند، در حالی که صحت می‌تواند متفاوت باشد. صحت و "),
  en("F1"), fa(" کلان نیز محاسبه و ثبت شده‌اند."),
], { indent: { firstLine: 0 } }));

metrics.push(P([fa(
  "هیچ صحتی بدون پایه‌ی خود گزارش نمی‌شود. در بخشی با توزیعِ نامتوازن، " +
  "پاسخ‌دادنِ همیشگیِ کلاسِ پرتکرار صحتی می‌دهد که می‌تواند محترمانه به " +
  "نظر برسد؛ عددی که آن پایه را همراه نداشته باشد خواننده را گمراه می‌کند."),
]));

// ---- goes into §5, the raw differences ----
const perSeed = [];
perSeed.push(H("۵-۹ تفاضل‌های خام", 2));
perSeed.push(P([fa(
  "جدول ("), en("9"), fa(
  ") تفاضلِ هر بذر را بدون خلاصه‌سازی می‌آورد. میانگین و انحراف معیار " +
  "می‌توانند الگویی را پنهان کنند که در پنج عدد آشکار است — مثلاً این‌که " +
  "همه‌ی بذرها هم‌علامت‌اند یا یک بذرِ پرت تمام اثر را ساخته است."),
], { indent: { firstLine: 0 } }));

perSeed.push(caption(
  "جدول (۹): پراکندگیِ خامِ سودِ ترکیب در پانزده بذر"));
const W9 = [1500, 800, 700, 700, 700];
perSeed.push(table(W9, [
  new TableRow({ tableHeader: true, children: [
    cell([T("رمزگذار", { bold: true })], { header: true, width: W9[0] }),
    cell([T("هم‌علامت", { bold: true })], { header: true, width: W9[1] }),
    cell([T("کمینه", { bold: true })], { header: true, width: W9[2] }),
    cell([T("بیشینه", { bold: true })], { header: true, width: W9[3] }),
    cell([T("میانه", { bold: true })], { header: true, width: W9[4] }),
  ]}),
  ...ENC.map((e) => {
    const d = ben(e).per_seed;
    const sorted = [...d].sort((x, y) => x - y);
    const positive = d.filter((x) => x > 0).length;
    // How many seeds share the sign of the mean. A table of fifteen raw
    // differences does not fit a two-column page, and this answers the
    // question that table was there for: whether one lucky run carries the
    // effect.
    const agreeing = ben(e).mean >= 0 ? positive : d.length - positive;
    const median = sorted[Math.floor(sorted.length / 2)];
    return new TableRow({ children: [
      cell([TL(e)], { width: W9[0], align: AlignmentType.START }),
      cell([TL(`${agreeing}/${d.length}`)], { width: W9[1], ltr: true }),
      cell([TL(signed(sorted[0]))], { width: W9[2], ltr: true }),
      cell([TL(signed(sorted[sorted.length - 1]))], { width: W9[3], ltr: true }),
      cell([TL(signed(median))], { width: W9[4], ltr: true }),
    ]});
  }),
]));

perSeed.push(P([fa(
  "ستونِ «هم‌علامت» می‌گوید چند بذر از پانزده با علامتِ میانگین هم‌جهت‌اند. " +
  "اگر اثری را یک اجرای خوش‌شانس می‌ساخت، این ستون نزدیک نیمه می‌ایستاد و " +
  "فاصله‌ی کمینه تا بیشینه از صفر می‌گذشت. برای رمزگذارهایی که اثرشان " +
  "معنادار است چنین نیست."),
], { spacing: { before: 140 } }));

// ---- goes into §5, the multiplicity correction ----
// Fourteen paired tests carry the results section. A reader is entitled to
// ask what survives a correction, and a paper that does not answer invites
// the question at review. The family is fixed in scripts/multiplicity.py and
// read from runs/paper_numbers.json, so no threshold here is typed.
const multiplicity = [];
const MU = N.multiplicity;
if (MU && MU.tests) {
  const SHORT = {
    "CLIP ViT-L/14": "L/14",
    "CLIP ViT-B/32": "B/32",
    "jina-clip-v2": "jina",
    "CLIPfa": "CLIPfa",
  };
  const KIND = {
    "fusion benefit": "سود ترکیب",
    "architecture": "معماری",
    "interaction": "برهم‌کنش",
  };
  const rows = Object.entries(MU.tests)
    .sort((a, b) => a[1].rank - b[1].rank);

  multiplicity.push(H("۵-۱۰ تصحیح برای مقایسه‌های چندگانه", 2));
  multiplicity.push(P([fa(
    "چهارده آزمونِ جفتی بارِ این بخش را می‌برند: چهار سودِ ترکیب، چهار اثرِ " +
    "معماری و شش برهم‌کنش. در سطحِ "), en(ltr("α = 0.05")), fa(
    " چنین خانواده‌ای به‌تنهایی از شانس حدود "), en(ltr("0.7")), fa(
    " ردِ کاذب انتظار دارد، پس این پرسش که پس از تصحیح چه می‌ماند، پرسشِ " +
    "بجایی است. روشِ هولم–بونفرونی به کار رفته است: هیچ فرضی درباره‌ی " +
    "وابستگیِ آزمون‌ها نمی‌گذارد و به‌طور یکنواخت از بونفرونیِ ساده " +
    "تواناتر است. خانواده پیش از دیدنِ آستانه‌ها در "),
    en("scripts/multiplicity.py"), fa(
    " تثبیت شده است، تا هیچ ادعایی نتواند برای نجاتِ خود از خانواده بیرون " +
    "برود."),
  ], { indent: { firstLine: 0 } }));

  multiplicity.push(caption(
    "جدول (۱۰): هولم–بونفرونی روی چهارده ادعای بخش ۵، به ترتیبی که روش " +
    "بررسی می‌کند"));
  const W10 = [1320, 600, 700, 740, 900];
  multiplicity.push(table(W10, [
    new TableRow({ tableHeader: true, children: [
      cell([T("مقایسه", { bold: true })], { header: true, width: W10[0] }),
      cell([T("p", { bold: true })], { header: true, width: W10[1] }),
      cell([T("آستانه", { bold: true })], { header: true, width: W10[2] }),
      cell([T("حاشیه", { bold: true })], { header: true, width: W10[3] }),
      cell([T("", { bold: true })], { header: true, width: W10[4] }),
    ]}),
    ...rows.map(([label, row]) => {
      const [kind, pair] = label.split(": ");
      const names = pair.split(" − ").map((e) => SHORT[e] || e).join(" − ");
      const shown = row.p < 0.001 ? "<0.001" : fmt(row.p, 3);
      // cantSplit: fourteen rows cross the column break, and without it a
      // two-line label is torn in half -- "برهم‌کنش: B/32 −" ending one
      // column and "jina" opening the next.
      return new TableRow({ cantSplit: true, children: [
        cell([T(KIND[kind] || kind), TL(`: ${names}`)],
             { width: W10[0], align: AlignmentType.START }),
        cell([TL(shown, { bold: row.holds })], { width: W10[1], ltr: true }),
        cell([TL(fmt(row.threshold, 4))], { width: W10[2], ltr: true }),
        cell([TL(signed(row.margin, 4), { bold: row.holds && row.margin < 0.01 })],
             { width: W10[3], ltr: true }),
        cell([T(row.holds ? "می‌ماند" : "نمی‌ماند", { bold: !row.holds })],
             { width: W10[4] }),
      ]});
    }),
  ]));

  multiplicity.push(P([
    fa("هر ادعایی که این مقاله معنادار می‌خواند، می‌ماند.", { bold: true }),
    fa(" دو سطری که نمی‌مانند، دقیقاً همان دو سطری‌اند که مقاله آن‌ها را " +
    "«بی‌اثر» گزارش می‌کند: برهم‌کنشِ "), en("L/14"), fa(" و "), en("B/32"),
    fa(" که در ۵-۴ نه با "), en("p"), fa(
    " بلکه با آزمونِ هم‌ارزی برقرار می‌شود، و اثرِ معماری روی "), en("B/32"),
    fa(". تصحیح، ردکردن را سخت‌تر می‌کند، پس نمی‌تواند ادعای «تفاوتی نیست» " +
    "را تضعیف کند — و آزمون‌های هم‌ارزی به همین دلیل عمداً تصحیح‌نشده " +
    "می‌مانند."),
  ], { spacing: { before: 140 } }));

  const tight = MU.tightest ? MU.tests[MU.tightest] : null;
  if (tight) {
    multiplicity.push(P([fa(
      "ولی تصحیح برای خواندنِ یک ادعا بی‌اثر نیست. اثرِ "), en("MemeCLIP"),
      fa(" روی "), en("CLIP ViT-L/14"), fa(" با "),
      en(ltr(`p = ${fmt(tight.p, 3)}`)), fa(
      " باریک‌ترین حاشیه‌ی خانواده را دارد؛ ستونِ «حاشیه» در جدول آن را " +
      "می‌دهد. همین ادعاست که در چکیده آمده. می‌ماند، و در عینِ حال نخستین " +
      "چیزی است که انتظار می‌رود یک بازتولید از دست بدهد. مقاله آن را " +
      "گزارش می‌کند تا خواننده خودش وزنش را تعیین کند."),
    ]));
  }

  multiplicity.push(P([fa(
    "گستراندنِ خانواده به دوازده آزمونِ فاصله‌ی رمزگذارها — جمعاً ۲۶ آزمون — " +
    "هیچ‌یک از این‌ها را عوض نمی‌کند: همان دوازده می‌مانند و همان دو " +
    "نمی‌مانند."),
  ]));
}

// ---- goes into §6 ----
const proxy = [];
proxy.push(H("۳-۳ چرا بازیابی سنجه‌ی درستی برای هم‌ترازی است", 2));
proxy.push(P([fa(
  "بازیابیِ صفر-شات دقیقاً همان چیزی را می‌سنجد که ترکیب عنصری به آن " +
  "نیاز دارد: این‌که آیا نمایشِ یک متن و نمایشِ تصویرِ متناظرش در یک فضا " +
  "به هم نزدیک‌اند. هیچ برچسبی لازم ندارد، هیچ آموزشی در کار نیست، و " +
  "بنابراین نمی‌تواند از همان داده‌ای که می‌خواهیم توضیح دهیم اطلاع بگیرد. " +
  "این استقلال است که اجازه می‌دهد آن را «پیش‌بین» بنامیم و نه «هم‌بسته»."),
], { indent: { firstLine: 0 } }));

proxy.push(P([fa(
  "با این حال سنجه‌ای کامل نیست و دو محدودیتش را باید گفت. نخست، بازیابی " +
  "کیفیتِ رتبه‌بندیِ کلی را می‌سنجد و نه هم‌ترازیِ بُعد‌به‌بُعد را، که آن " +
  "چیزی است که ضرب عنصری واقعاً از آن استفاده می‌کند؛ این دو مرتبط‌اند " +
  "ولی یکی نیستند. دوم، زیرنویسِ توصیفیِ یک تصویر با متنِ یک میم هم‌جنس " +
  "نیست: اولی تصویر را توصیف می‌کند و دومی با آن بازی می‌کند. پس آنچه " +
  "اندازه گرفته می‌شود کرانِ بالایی از هم‌ترازیِ قابل‌دسترس در آن زبان " +
  "است، نه هم‌ترازی در خودِ وظیفه."),
]));

// ---- ethics ----
const ethics = [];
ethics.push(H("۱۰- ملاحظات اخلاقی", 1));
ethics.push(P([fa(
  "داده‌ی این حوزه محتوای آزارنده است و برچسب‌هایش قضاوتِ انسانی. پیکره‌ی " +
  "مرجع برچسبِ خود را از حاشیه‌نویسانی در یک بافتِ فرهنگی گرفته است، و "),
  en("Multi3Hate"), fa(
  " نشان می‌دهد که پنج کشور بر سرِ یک میمِ یکسان اختلاف نظر دارند. " +
  "بنابراین «نفرت‌پراکن» در این مقاله به‌معنای قضاوتِ همان حاشیه‌نویسان است " +
  "و نه حقیقتی جهان‌شمول."),
], { indent: { firstLine: 0 } }));

ethics.push(P([fa(
  "این نکته برای فارسی اهمیتِ عملی دارد. هر مجموعه‌ای که در آینده ساخته " +
  "شود، مرزِ میان نقدِ یک نهاد و حمله به یک هویت را باید صریح تعریف کند، " +
  "وگرنه رده‌بندی که بر آن آموزش می‌بیند می‌تواند به ابزارِ خاموش‌کردنِ " +
  "سخن بدل شود. پروتکلِ پیشنهادیِ ما این مرز را به‌صورت مکتوب دارد. " +
  "همچنین هیچ تصویری در این پژوهش بازنشر نشده و منشأ هر نمونه در زمانِ " +
  "گردآوری ثبت می‌شود، نه پس از آن."),
]));

module.exports = {
  related, method, metrics, perSeed, multiplicity, proxy, ethics,
};
