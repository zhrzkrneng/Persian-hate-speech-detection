// Section 6, rewritten. The four-encoder result made a prediction; the
// twenty-cell extension tested it and it did not hold. That is what this
// section reports.
const B = require("./build.js");
const { P, H, caption, cell, table, T, TL, fmt, signed, ltr, N, fs,
        TableRow, Paragraph, TextRun, ImageRun, AlignmentType, fa, en, SIZE } = B;

const M = N.multilingual || {};
const cells = M.cells || [];
const over = M.overall || {};
const subs = M.subsets || {};
const perEnc = M.per_encoder || {};
const star = (p) => (p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : "n.s.");

const body = [];

body.push(H("۶- آزمودنِ پیش‌بینی: افزونه‌ی چندزبانه", 1));
body.push(P([fa(
  "نتیجه‌ی بخش ۵ یک پیش‌بینیِ آزمودنی می‌سازد: اگر سودِ ترکیب تابع " +
  "هم‌ترازیِ رمزگذار باشد، باید در هر جفتِ (رمزگذار، زبان) از هم‌ترازیِ " +
  "همان زبان قابل پیش‌بینی باشد، نه فقط در چهار نقطه‌ی یک پیکره‌ی " +
  "انگلیسی. این بخش آن را می‌آزماید و گزارش می‌کند که "),
  fa("برقرار نشد", { bold: true }), fa("."),
], { indent: { firstLine: 0 } }));

body.push(P([fa("طرح آزمون: سری که روی "), en("Hateful Memes"), fa(
  " آموزش دیده، بدون هیچ آموزش بیشتری، ۳۰۰ میمِ موازیِ "), en("Multi3Hate"),
  fa(" ["), en("5"), fa(
  "] را در پنج زبان می‌خواند. میم‌ها موازی‌اند — همان تصویر و همان معنا، " +
  "تنها با تغییر متنِ روی تصویر — و هر زبان برچسبِ کشورِ بومی‌اش را " +
  "می‌گیرد. هر سلول پنج بذر دارد، و هم‌ترازیِ همان زبان از "),
  en("Crossmodal-3600"), fa(" برداشته می‌شود. چهار رمزگذار در پنج زبان، " +
  "بیست نقطه."),
]));

if (cells.length) {
  body.push(caption(
    "جدول (۱۱): بیست سلولِ صفر-شات. سودِ ترکیب تفاضلِ جفتی روی پنج بذر است"));
  const W = [1250, 500, 800, 750, 800, 500];
  body.push(table(W, [
    new TableRow({ tableHeader: true, children: [
      cell([T("رمزگذار", { bold: true })], { header: true, width: W[0] }),
      cell([T("زبان", { bold: true })], { header: true, width: W[1] }),
      cell([TL("align R@1", { bold: true })], { header: true, width: W[2] }),
      cell([TL("concat", { bold: true })], { header: true, width: W[3] }),
      cell([T("سود", { bold: true })], { header: true, width: W[4] }),
      cell([TL("p", { bold: true })], { header: true, width: W[5] }),
    ]}),
    ...cells.map((r) => new TableRow({ children: [
      cell([TL(r.encoder)], { width: W[0], align: AlignmentType.START }),
      cell([TL(r.language)], { width: W[1], ltr: true }),
      cell([TL(fmt(r.alignment))], { width: W[2], ltr: true }),
      cell([TL(fmt(r.concat))], { width: W[3], ltr: true }),
      cell([TL(signed(r.benefit), { bold: r.benefit_p < 0.05 })],
        { width: W[4], ltr: true }),
      cell([TL(fmt(r.benefit_p, 3))], { width: W[5], ltr: true }),
    ]})),
  ]));
}

body.push(H("۶-۱ رابطه برقرار نیست", 2));
body.push(P([fa("روی هر بیست سلول، هم‌ترازی سودِ ترکیب را پیش‌بینی نمی‌کند: "),
  en(ltr(`r = ${fmt(over.pearson_r, 3)}`)), fa(" ("),
  en(ltr(`p = ${fmt(over.pearson_p, 3)}`)), fa(") و ضریب رتبه‌ای "),
  en(ltr(`rho = ${fmt(over.spearman_rho, 3)}`)), fa(" ("),
  en(ltr(`p = ${fmt(over.spearman_p, 3)}`)), fa(
  "). این با چهار نقطه‌ی بخش ۵ که علامت عوض می‌کردند، در تضاد نیست — " +
  "بلکه نشان می‌دهد آن الگو به این محور تعمیم نمی‌یابد."),
], { indent: { firstLine: 0 } }));

const floors = Object.keys(subs).sort();
if (floors.length) {
  body.push(P([fa(
    "نخستین اعتراض به هر نتیجه‌ی منفی، اثرِ کف است: بسیاری از این سلول‌ها " +
    "نزدیک "), en("AUROC"), fa(
    " پنجاه‌اند، و تفاضلِ دو رده‌بندِ تقریباً تصادفی چیزی نمی‌سنجد. پس " +
    "به‌جای استدلال، محاسبه شد:"),
  ]));
  body.push(caption("جدول (۱۲): همان همبستگی، با حذف سلول‌های نزدیک شانس"));
  const W = [1800, 500, 1150, 1150];
  body.push(table(W, [
    new TableRow({ tableHeader: true, children: [
      cell([T("زیرمجموعه", { bold: true })], { header: true, width: W[0] }),
      cell([TL("n", { bold: true })], { header: true, width: W[1] }),
      cell([TL("Pearson r", { bold: true })], { header: true, width: W[2] }),
      cell([TL("Spearman rho", { bold: true })], { header: true, width: W[3] }),
    ]}),
    new TableRow({ children: [
      cell([T("همه‌ی سلول‌ها")], { width: W[0], align: AlignmentType.START }),
      cell([TL(String(over.n))], { width: W[1], ltr: true }),
      cell([TL(`${fmt(over.pearson_r, 3)} (p=${fmt(over.pearson_p, 2)})`)],
        { width: W[2], ltr: true }),
      cell([TL(`${fmt(over.spearman_rho, 3)} (p=${fmt(over.spearman_p, 2)})`)],
        { width: W[3], ltr: true }),
    ]}),
    ...floors.map((f) => new TableRow({ children: [
      cell([TL(`concat AUROC >= ${Math.round(Number(f))}`)],
        { width: W[0], align: AlignmentType.START, ltr: true }),
      cell([TL(String(subs[f].n))], { width: W[1], ltr: true }),
      cell([TL(`${fmt(subs[f].pearson_r, 3)} (p=${fmt(subs[f].pearson_p, 2)})`)],
        { width: W[2], ltr: true }),
      cell([TL(`${fmt(subs[f].spearman_rho, 3)} (p=${fmt(subs[f].spearman_p, 2)})`)],
        { width: W[3], ltr: true }),
    ]})),
  ]));
  body.push(P([fa(
    "حذف سلول‌های ضعیف رابطه‌ای نمی‌سازد. اثرِ کف توضیحش نمی‌دهد."),
  ], { spacing: { before: 140 } }));
}

body.push(H("۶-۲ آنچه تعمیم یافت", 2));
if (Object.keys(perEnc).length) {
  body.push(P([fa(
    "اگر سودِ ترکیب را به‌ازای رمزگذار و روی پنج زبان میانگین بگیریم، " +
    "الگوی بخش ۵ دوباره ظاهر می‌شود:"),
  ], { indent: { firstLine: 0 } }));
  body.push(caption("جدول (۱۳): سودِ ترکیب به‌ازای رمزگذار، میانگین روی پنج زبان"));
  const W = [1500, 800, 700, 700, 900];
  const order = ["CLIP ViT-L/14", "CLIP ViT-B/32", "CLIPfa", "jina-clip-v2"];
  body.push(table(W, [
    new TableRow({ tableHeader: true, children: [
      cell([T("رمزگذار", { bold: true })], { header: true, width: W[0] }),
      cell([T("میانگین", { bold: true })], { header: true, width: W[1] }),
      cell([TL("sd", { bold: true })], { header: true, width: W[2] }),
      cell([TL("p", { bold: true })], { header: true, width: W[3] }),
      cell([T("منفی در", { bold: true })], { header: true, width: W[4] }),
    ]}),
    ...order.filter((e) => perEnc[e]).map((e) => new TableRow({ children: [
      cell([TL(e)], { width: W[0], align: AlignmentType.START }),
      cell([TL(signed(perEnc[e].mean), { bold: perEnc[e].p < 0.05 })],
        { width: W[1], ltr: true }),
      cell([TL(fmt(perEnc[e].sd))], { width: W[2], ltr: true }),
      cell([TL(`${fmt(perEnc[e].p, 3)} ${star(perEnc[e].p)}`)],
        { width: W[3], ltr: true }),
      cell([TL(`${perEnc[e].negative_in}/${perEnc[e].n}`)],
        { width: W[4], ltr: true }),
    ]})),
  ]));
  const j = perEnc["jina-clip-v2"];
  if (j) {
    body.push(P([en("jina-clip-v2"), fa(" در "),
      en(ltr(`${j.negative_in}/${j.n}`)), fa(" زبان از ترکیب عنصری "),
      fa("ضرر", { bold: true }), fa(" می‌کند ("),
      en(ltr(`${signed(j.mean)}، p = ${fmt(j.p, 3)}`)), fa(
      ") در حالی که در هر پنج زبان بالاترین هم‌ترازی را دارد. این با " +
      "نتیجه‌ی آن روی "), en("Hateful Memes"), fa(
      " سازگار است و با این‌که هم‌ترازی تعیین‌کننده باشد، نیست. یعنی پدیده " +
      "واقعی است و وابسته به رمزگذار؛ آنچه آزموده و رد شد، "),
      fa("توضیحِ", { bold: true }), fa(" آن است."),
    ], { spacing: { before: 140 } }));
  }
}

const fig = "multilingual.png";
if (fs.existsSync(fig)) {
  body.push(new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 160, after: 40 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(fig),
      transformation: { width: 300, height: 190 } })],
  }));
  body.push(caption(
    "شکل (۲): بیست سلولِ (رمزگذار، زبان). محور افقی هم‌ترازیِ اندازه‌گیری‌شده " +
    "در همان زبان (R@1 بازیابی متن به تصویر، درصد)، محور عمودی سودِ ترکیب " +
    "(اختلاف AUROC صفر-شات میان align و concat). میله‌ها خطای معیار روی " +
    "پنج بذر. خط‌چین افقی «بدون سود» را نشان می‌دهد."));
}

body.push(H("۶-۳ چرا این همان کمیت نیست", 2));
body.push(P([fa(
  "دو تفاوت میان این بیست سلول و چهار سلولِ بخش ۵ هست که نتیجه را " +
  "مشروط می‌کند و باید صریح گفته شود."),
], { indent: { firstLine: 0 } }));
body.push(P([fa("نخست، این‌ها "), fa("انتقالِ صفر-شات", { bold: true }), fa(
  " هستند. سر روی پیکره‌ی انگلیسی آموزش می‌بیند و پیکره‌ای دیگر را در " +
  "زبانی دیگر می‌خواند، در حالی که سودِ بخش ۵ درون‌توزیع اندازه‌گیری شده. " +
  "سطحِ مطلق اینجا بسیار پایین‌تر است و در چند سلول نزدیک شانس."),
]));
body.push(P([fa("دوم، محور "), fa("میان‌زبانی", { bold: true }), fa(
  " است. زیرنویس‌های "), en("Crossmodal-3600"), fa(
  " برای هر زبان جداگانه نوشته شده‌اند، نه ترجمه؛ آلمانی برای همان تصاویر " +
  "زیرنویس‌های بیشتر و بلندتری دارد و پرس‌وجوی بلندتر، بازیابیِ آسان‌تر. " +
  "پس بخشی از فاصله‌ی میان‌زبانی سبکِ حاشیه‌نویسی است و نه توانِ رمزگذار. " +
  "مقایسه‌ی رمزگذارها "), fa("درونِ", { italics: true }), fa(
  " یک زبان از این آلودگی مصون است؛ مقایسه‌ی زبان‌ها نیست، و این محور " +
  "دقیقاً بر همان بنا شده."),
]));

body.push(H("۶-۴ فارسی", 2));
body.push(P([fa(
  "هیچ مجموعه‌ی عمومیِ میمِ نفرت‌پراکنِ فارسی وجود ندارد و این پژوهش " +
  "حاشیه‌نویسی در اختیار ندارد که یکی بسازد، پس هیچ عددی برای فارسی " +
  "گزارش نمی‌شود. آنچه پیش از این آزمون می‌شد گفت، پیش‌بینی از روی " +
  "هم‌ترازیِ فارسی بود. آن آزمون انجام شد و پشتیبانی نکرد، پس آن پیش‌بینی " +
  "هم پس گرفته می‌شود."),
], { indent: { firstLine: 0 } }));
body.push(P([fa(
  "آنچه می‌ماند ضعیف‌تر و صادقانه‌تر است. برای فارسی، انتخابِ رمزگذار " +
  "اهمیتی به‌مراتب بیشتر از انتخابِ سازوکار ترکیب دارد — پراکندگیِ " +
  "میان‌رمزگذاری در جدول ۱ چند برابرِ هر اثر ترکیبی است، و خانواده‌ی "),
  en("CLIP"), fa(
  " روی فارسی در سطح شانس بازیابی می‌کند. و چون سودِ ترکیب به رمزگذار " +
  "وابسته است و نه به هم‌ترازی، هیچ قاعده‌ی سرانگشتی‌ای نمی‌گوید کدام " +
  "ترکیب برای یک رمزگذارِ فارسی بهتر است: باید سنجیده شود. آزمودنِ هر دو " +
  "گزاره به مجموعه‌ای نیاز دارد که هنوز ساخته نشده، و پروتکلِ ساختش " +
  "به‌عنوان بخشی از مخزن منتشر شده است."),
]));

module.exports = { body };
