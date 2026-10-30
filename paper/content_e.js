// Sections that use data already collected but not yet shown: the architecture
// comparison, the encoder gap, the validation/test disagreement, the
// tokenisation table, and reproducibility.
const B = require("./build.js");
const F = require("./content_f.js");
const { P, H, caption, cell, table, T, TL, fmt, signed, ltr, N,
        TableRow, Paragraph, TextRun, AlignmentType, fa, en, SIZE } = B;

const ENC = ["CLIP ViT-L/14", "CLIP ViT-B/32", "jina-clip-v2", "CLIPfa"];
const star = (p) => (p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : "n.s.");
const arch = (e) => N.architecture[e];
const gap = (k) => N.encoder_gap[k];
const cells = (e, k) => N.cells[e][k];

const afterResults = [];

// ---------- architecture ----------
afterResults.push(H("۵-۶ کدام معماری برنده است هم به رمزگذار بستگی دارد", 2));
afterResults.push(P([en("MemeCLIP"), fa(
  " سه چیز به ترکیب عنصری می‌افزاید: وفق‌دهنده‌های پس‌ماندی، نرمال‌سازی "),
  en("L2"), fa(
  " پیش از ترکیب، و رده‌بندِ کسینوسی. جدول ("), en("5"), fa(
  ") نشان می‌دهد این افزوده‌ها روی هر رمزگذار چه می‌کنند."),
], { indent: { firstLine: 0 } }));

afterResults.push(caption("جدول (۵): اثر معماری — MemeCLIP منهای Hate-CLIPper، هر دو با ترکیب عنصری"));
const W5 = [1500, 900, 800, 800];
afterResults.push(table(W5, [
  new TableRow({ tableHeader: true, children: [
    cell([T("رمزگذار", { bold: true })], { header: true, width: W5[0] }),
    cell([T("اختلاف", { bold: true })], { header: true, width: W5[1] }),
    cell([TL("sd", { bold: true })], { header: true, width: W5[2] }),
    cell([TL("p", { bold: true })], { header: true, width: W5[3] }),
  ]}),
  ...ENC.map((e) => new TableRow({ children: [
    cell([TL(e)], { width: W5[0], align: AlignmentType.START }),
    cell([TL(signed(arch(e).mean))], { width: W5[1], ltr: true }),
    cell([TL(fmt(arch(e).sd))], { width: W5[2], ltr: true }),
    cell([TL(`${fmt(arch(e).p, 3)} ${star(arch(e).p)}`)], { width: W5[3], ltr: true }),
  ]})),
]));

afterResults.push(P([fa("علامت عوض می‌شود. همان سه افزوده روی "),
  en("CLIP ViT-L/14"), fa(" کمک می‌کنند ("),
  en(ltr(`${signed(arch("CLIP ViT-L/14").mean)}، p = ${fmt(arch("CLIP ViT-L/14").p, 3)}`)),
  fa(") و روی "), en("jina-clip-v2"), fa(" آسیب می‌زنند ("),
  en(ltr(`${signed(arch("jina-clip-v2").mean)}، p = ${fmt(arch("jina-clip-v2").p, 3)}`)),
  fa(
  "). این هم‌راستا با یافته‌ی مرکزی است: برتریِ یک معماری که تنها روی یک " +
  "رمزگذار گزارش شده باشد، خاصیتِ آن جفت است نه خاصیتِ معماری."),
], { spacing: { before: 140 } }));

// ---------- encoder gap ----------
afterResults.push(H("۵-۷ فاصله‌ی میان رمزگذارها", 2));
afterResults.push(P([fa(
  "برای آن‌که اندازه‌ی نسبیِ اثرها روشن باشد، جدول ("), en("6"), fa(
  ") فاصله‌ی امتیاز میان رمزگذارها را زیر الحاق می‌آورد."),
], { indent: { firstLine: 0 } }));

afterResults.push(caption("جدول (۶): فاصله‌ی AUROC میان رمزگذارها، زیر ترکیب الحاقی"));
const W6 = [2450, 850, 700, 600];
const gapKeys = Object.keys(N.encoder_gap).filter((k) => k.endsWith("(concat)"));
afterResults.push(table(W6, [
  new TableRow({ tableHeader: true, children: [
    cell([T("زوج", { bold: true })], { header: true, width: W6[0] }),
    cell([T("اختلاف", { bold: true })], { header: true, width: W6[1] }),
    cell([TL("sd", { bold: true })], { header: true, width: W6[2] }),
    cell([TL("p", { bold: true })], { header: true, width: W6[3] }),
  ]}),
  ...gapKeys.map((k) => new TableRow({ children: [
    cell([TL(k.replace(" (concat)", "").replace(" - ", " − "))],
      { width: W6[0], align: AlignmentType.START }),
    cell([TL(signed(gap(k).mean))], { width: W6[1], ltr: true }),
    cell([TL(fmt(gap(k).sd))], { width: W6[2], ltr: true }),
    cell([TL(`${fmt(gap(k).p, 3)} ${star(gap(k).p)}`)], { width: W6[3], ltr: true }),
  ]})),
]));

afterResults.push(P([fa(
  "بزرگ‌ترین فاصله چندین برابرِ بزرگ‌ترین اثرِ ترکیب یا معماری است. برای " +
  "کسی که سامانه‌ای می‌سازد، این ترتیبِ اولویت است: نخست رمزگذار، سپس " +
  "همه‌ی چیزهای دیگر."),
], { spacing: { before: 140 } }));

// ---------- validation vs test ----------
afterResults.push(H("۵-۸ چرا ستون اعتبارسنجی مقایسه نمی‌کند", 2));
const vrow = (e, k) => cells(e, k).val_auroc;
afterResults.push(P([fa(
  "جدول ("), en("7"), fa(
  ") همان دوازده سلول را با ستونِ اعتبارسنجی کنار ستونِ آزمون می‌گذارد. " +
  "این دو گاهی بر سرِ جهتِ ترکیب با هم اختلاف دارند، و این اختلاف را " +
  "گزارش می‌کنیم چون پنهان‌کردنش بدتر است."),
], { indent: { firstLine: 0 } }));

afterResults.push(caption("جدول (۷): AUROC اعتبارسنجی در برابر آزمون، ترکیب عنصری"));
const W7 = [1500, 1150, 1150];
afterResults.push(table(W7, [
  new TableRow({ tableHeader: true, children: [
    cell([T("رمزگذار", { bold: true })], { header: true, width: W7[0] }),
    cell([T("اعتبارسنجی", { bold: true })], { header: true, width: W7[1] }),
    cell([T("آزمون", { bold: true })], { header: true, width: W7[2] }),
  ]}),
  ...ENC.map((e) => new TableRow({ children: [
    cell([TL(e)], { width: W7[0], align: AlignmentType.START }),
    cell([TL(`${fmt(vrow(e, "hateclipper/align").mean)} ± ${fmt(vrow(e, "hateclipper/align").sd)}`)],
      { width: W7[1], ltr: true }),
    cell([TL(`${fmt(cells(e, "hateclipper/align").test.auroc.mean)} ± ${fmt(cells(e, "hateclipper/align").test.auroc.sd)}`)],
      { width: W7[2], ltr: true }),
  ]})),
]));

afterResults.push(P([fa(
  "عددِ اعتبارسنجیِ هر اجرا بیشینه‌ی بیست دوره است که انتخابِ مدل برداشته " +
  "است. یک بیشینه‌ی انتخاب‌شده اریبِ رو به بالا دارد، و هیچ تضمینی نیست که " +
  "این اریب میان دو گونه برابر باشد: "), en("align"), fa(" به سر "), en("n"),
  fa(" ورودی می‌دهد و "), en("concat"), fa(" مقدار "), en("2n"), fa(
  "، پس توانِ متفاوتی برای برازش بر پانصد نمونه دارند و انتظار می‌رود " +
  "متفاوت هم بر آن اوج بگیرند. بنابراین اختلافِ دو بیشینه‌ی انتخاب‌شده، " +
  "اختلاف در این است که هر گونه چقدر مطمئن به یک نقطه‌ی بلندِ اعتبارسنجی " +
  "می‌رسد، نه اختلاف در آنچه تعمیم می‌دهد. بخش آزمون یک‌بار خوانده می‌شود " +
  "و تنها ستونی است که مقایسه می‌کند."),
], { spacing: { before: 140 } }));

afterResults.push(...F.perSeed);

// ---------- tokenisation table ----------
const tokRows = (N.tokenisation.rows || []);
const afterTok = [];
if (tokRows.length) {
  afterTok.push(caption("جدول (۸): هزینه‌ی توکن‌سازی برای زیرنویس‌هایی از همان تصاویر"));
  const W8 = [1700, 500, 800, 700, 900];
  afterTok.push(table(W8, [
    new TableRow({ tableHeader: true, children: [
      cell([T("مدل", { bold: true })], { header: true, width: W8[0] }),
      cell([T("زبان", { bold: true })], { header: true, width: W8[1] }),
      cell([T("میانگین", { bold: true })], { header: true, width: W8[2] }),
      cell([T("بیشینه", { bold: true })], { header: true, width: W8[3] }),
      cell([T("بریده‌شده", { bold: true })], { header: true, width: W8[4] }),
    ]}),
    ...tokRows.map((row) => new TableRow({ children: [
      cell([TL(row.model.split("/")[1] || row.model)],
        { width: W8[0], align: AlignmentType.START }),
      cell([TL(row.language)], { width: W8[1], ltr: true }),
      cell([TL(fmt(row.mean_tokens))], { width: W8[2], ltr: true }),
      cell([TL(String(row.max_tokens))], { width: W8[3], ltr: true }),
      cell([TL(`${fmt(row.over_clip_context_pct, 1)}%`)], { width: W8[4], ltr: true }),
    ]})),
  ]));
}

// ---------- reproducibility ----------
const repro = [];
repro.push(H("۹- تکرارپذیری", 1));
repro.push(P([fa(
  "هر عددی که در این مقاله آمده از پرونده‌های نتیجه‌ی به‌ازای بذر تولید " +
  "شده است، نه با تایپ دستی: متنِ مقاله از همان پرونده‌هایی ساخته می‌شود " +
  "که اجراها نوشته‌اند، پس یک ادعا در متن و آماره‌ی پشتش نمی‌توانند از هم " +
  "جدا بیفتند. ویژگی‌های رمزگذار یک‌بار محاسبه و با ثبتِ منشأ ذخیره " +
  "می‌شوند — نام رمزگذار، بخش داده، تعداد نمونه، پهنای ویژگی، و نسخه‌ی " +
  "کتابخانه‌ها — و آموزش، کَشی را که رمزگذار یا نسخه‌ی ثبت‌شده‌اش با " +
  "پیکربندی نخواند رد می‌کند."),
], { indent: { firstLine: 0 } }));

repro.push(P([fa(
  "هیچ پرونده‌ی داده‌ای در مخزن قرار نگرفته است. اسکریپتِ تهیه‌ی "),
  en("Hateful Memes"), fa(
  " چیزی دانلود نمی‌کند؛ تنها نسخه‌ی محلیِ کاربر را وارسی می‌کند و " +
  "بررسی می‌کند که همه‌ی تصاویرِ ارجاع‌شده واقعاً موجود باشند — آزمونی که " +
  "لازم بود، چون دو نسخه‌ی آینه‌ایِ عمومی از این پیکره تعداد سطرِ درست و " +
  "دو هزار تصویرِ کم دارند و بارگیری‌شان بدون خطا تمام می‌شود."),
]));

repro.push(...F.ethics);

module.exports = { afterResults, afterTok, repro };
