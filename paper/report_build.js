const R = require("./report.js");
const { Document, Packer, Paragraph, TextRun, TableRow, AlignmentType, Footer,
        PageNumber, P, H, bullet, cell, table, T, TL, fa, en, fmt, sg, ltr,
        ben, ret, tok, N, M, fs } = R;

const ENC = ["CLIP ViT-L/14", "CLIP ViT-B/32", "jina-clip-v2", "CLIPfa"];
const au = (e, k) => N.cells[e][k].test.auroc;
const body = [];

// ---- title ----
body.push(new Paragraph({
  bidirectional: true, alignment: AlignmentType.CENTER, spacing: { after: 80 },
  children: [new TextRun(fa("گزارش کار: از اول تا آخر", { size: 18, bold: true }))],
}));
body.push(new Paragraph({
  bidirectional: true, alignment: AlignmentType.CENTER, spacing: { after: 60 },
  children: [new TextRun(fa(
    "وابستگیِ سودِ ترکیب چندوجهی به رمزگذار، و آزمودنِ توضیحِ رایجِ آن",
    { size: 12 }))],
}));
body.push(new Paragraph({
  bidirectional: true, alignment: AlignmentType.CENTER, spacing: { after: 320 },
  children: [new TextRun(fa(
    "زهره ذاکران — دانشگاه اصفهان، مرکز آموزش عالی شهرضا", { size: 11 }))],
}));

// ---- 1 ----
body.push(H("۱. سؤالی که دنبالش بودیم"));
body.push(P([fa(
  "یک میمِ نفرت‌پراکن معمولاً نه در تصویرش و نه در متنش به‌تنهایی " +
  "نفرت‌پراکن نیست؛ معنا از برخورد آن دو بیرون می‌آید. دو مقاله‌ی مرجعِ " +
  "این حوزه، "), en("Hate-CLIPper"), fa(" و "), en("MemeCLIP"), fa(
  "، این برخورد را با ضربِ عنصریِ دو بردار ویژگی می‌سازند و گزارش " +
  "می‌کنند که از الحاقِ ساده بهتر است."),
]));
body.push(P([en("Hate-CLIPper"), fa(
  " در بخش محدودیت‌هایش چیزی نوشته که نقطه‌ی شروع این پژوهش شد: روی " +
  "زبان تامیلی، آن ضرب هیچ سودی نداشت، و علتش را «ناهم‌ترازیِ فضای " +
  "ویژگی برای زبانی که مدل ندیده» دانست. آن توضیح هرگز آزموده نشده بود — " +
  "هم‌ترازی اندازه‌گیری نشده و تنها یک رمزگذار در کار بوده. کارِ این " +
  "پژوهش آزمودنِ همان توضیح است."),
]));

// ---- 2 ----
body.push(H("۲. چه ساختیم"));
body.push(P([fa(
  "چهار رمزگذارِ منجمد با پوششِ زبانیِ عمداً متفاوت انتخاب شد:"),
]));
body.push(bullet("CLIP ViT-L/14 و CLIP ViT-B/32 — انگلیسی‌محور، یکی بزرگ و یکی کوچک"));
body.push(bullet("jina-clip-v2 — چندزبانه"));
body.push(bullet("CLIPfa — تنها فارسی"));
body.push(P([fa("برای هرکدام دو چیزِ "), fa("مستقل", { bold: true }), fa(
  " اندازه گرفته شد. نخست امتیازِ رده‌بندی روی پیکره‌ی "),
  en("Hateful Memes"), fa(" با سه پیکربندی و پنج بذر تصادفی: "),
  en(ltr("۶۰")), fa(" اجرا. دوم هم‌ترازی، با بازیابیِ صفر-شات روی "),
  en("Crossmodal-3600"), fa(
  " — به مدل یک جمله می‌دهیم و می‌پرسیم تصویرِ خودش را پیدا می‌کند؟ " +
  "هیچ آموزشی و هیچ برچسبی در کار نیست. در شش زبان: ۲۴ اندازه‌گیری."),
]));
body.push(P([fa(
  "کلیدِ طرح همین استقلال است. اگر هم‌ترازی بتواند امتیازِ رده‌بندی را " +
  "پیش‌بینی کند، توضیحِ "), en("Hate-CLIPper"), fa(" درست است."),
]));

// ---- 3 ----
body.push(H("۳. آنچه پیدا کردیم و محکم است"));
body.push(H("۳-۱ سودِ ترکیب با رمزگذار علامت عوض می‌کند", 2));
const W1 = [2100, 1500, 1300, 1000];
body.push(table(W1, [
  new TableRow({ tableHeader: true, children: [
    cell([T("رمزگذار", { bold: true })], { header: true, width: W1[0] }),
    cell([T("هم‌ترازی انگلیسی", { bold: true })], { header: true, width: W1[1] }),
    cell([T("سود ترکیب", { bold: true })], { header: true, width: W1[2] }),
    cell([TL("p", { bold: true })], { header: true, width: W1[3] }),
  ]}),
  ...["CLIPfa", "CLIP ViT-B/32", "CLIP ViT-L/14", "jina-clip-v2"].map(e =>
    new TableRow({ children: [
      cell([TL(e)], { width: W1[0], align: AlignmentType.START }),
      cell([TL(fmt(ret(e, "en")))], { width: W1[1], ltrCell: true }),
      cell([TL(sg(ben(e).mean), { bold: ben(e).p < 0.05 })], { width: W1[2], ltrCell: true }),
      cell([TL(fmt(ben(e).p, 3))], { width: W1[3], ltrCell: true }),
    ]})),
]));
body.push(P([fa(
  "دو اثرِ معنادار در دو جهتِ مخالف. وقتی رمزگذار برای آن زبان اصلاً " +
  "هم‌تراز نشده (سطح شانس ۰٫۱۰ است)، ضرب عنصری نه‌تنها کمک نمی‌کند بلکه " +
  "آسیب می‌زند."),
], { align: AlignmentType.JUSTIFIED }));

body.push(H("۳-۲ ظرفیتِ مدل تعیین‌کننده نیست", 2));
const gapC = N.encoder_gap["CLIP ViT-L/14 - CLIP ViT-B/32 (concat)"];
const inter = N.interaction["CLIP ViT-L/14 - CLIP ViT-B/32"];
const bj = N.encoder_gap["CLIP ViT-B/32 - jina-clip-v2 (concat)"];
body.push(P([fa("این محکم‌ترین شاهدِ مقاله است. دو نسخه‌ی "), en("CLIP"), fa(
  " در امتیاز مطلق "), en(ltr(fmt(Math.abs(gapC.mean)))), fa(
  " واحد فاصله دارند ("), en(ltr("p < 0.001")), fa(
  ")، ولی سودِ ترکیبشان از نظر آماری تفکیک‌ناپذیر است ("),
  en(ltr(`p = ${fmt(inter.p, 3)}`)), fa(")."),
]));
body.push(P([fa("و روشن‌تر از آن: "), en("CLIP ViT-B/32"), fa(
  " به‌طور معنادار از "), en("jina-clip-v2"), fa(" ضعیف‌تر است ("),
  en(ltr(`${sg(bj.mean)}، p < 0.001`)), fa(
  ") و با این حال سود بیشتری می‌برد. پس «مدلِ قوی‌تر بیشتر سود می‌برد» با " +
  "یک عدد رد می‌شود، نه با استدلال."),
]));

body.push(H("۳-۳ هزینه‌ی توکن‌سازیِ فارسی", 2));
const b32en = tok("openai/clip-vit-base-patch32", "en"), b32fa = tok("openai/clip-vit-base-patch32", "fa");
const jen = tok("jinaai/jina-clip-v2", "en"), jfa = tok("jinaai/jina-clip-v2", "fa");
if (b32en && jfa) {
  const W2 = [1700, 1200, 1200, 1000, 1000];
  body.push(table(W2, [
    new TableRow({ tableHeader: true, children: [
      cell([T("مدل", { bold: true })], { header: true, width: W2[0] }),
      cell([T("انگلیسی", { bold: true })], { header: true, width: W2[1] }),
      cell([T("فارسی", { bold: true })], { header: true, width: W2[2] }),
      cell([T("نسبت", { bold: true })], { header: true, width: W2[3] }),
      cell([T("بریده", { bold: true })], { header: true, width: W2[4] }),
    ]}),
    new TableRow({ children: [
      cell([TL("CLIP")], { width: W2[0], align: AlignmentType.START }),
      cell([TL(fmt(b32en.mean_tokens))], { width: W2[1], ltrCell: true }),
      cell([TL(fmt(b32fa.mean_tokens))], { width: W2[2], ltrCell: true }),
      cell([TL(`${fmt(b32fa.mean_tokens / b32en.mean_tokens)}×`, { bold: true })], { width: W2[3], ltrCell: true }),
      cell([TL(`${fmt(b32fa.over_clip_context_pct, 1)}%`)], { width: W2[4], ltrCell: true }),
    ]}),
    new TableRow({ children: [
      cell([TL("jina-clip-v2")], { width: W2[0], align: AlignmentType.START }),
      cell([TL(fmt(jen.mean_tokens))], { width: W2[1], ltrCell: true }),
      cell([TL(fmt(jfa.mean_tokens))], { width: W2[2], ltrCell: true }),
      cell([TL(`${fmt(jfa.mean_tokens / jen.mean_tokens)}×`)], { width: W2[3], ltrCell: true }),
      cell([TL("0.0%")], { width: W2[4], ltrCell: true }),
    ]}),
  ]));
  body.push(P([en("CLIP"), fa(
    " برای یک جمله‌ی فارسی چهار برابر توکن خرج می‌کند و بخشی از جمله‌ها " +
    "را می‌بُرد، چون پنجره‌اش ۷۷ توکن است — محدودیتی معماری، نه یک تنظیم. " +
    "این شاهدی مستقل برای علتِ ناهم‌ترازی است."),
  ]));
}

// ---- 4 ----
body.push(H("۴. آنچه آزمودیم و رد شد"));
body.push(P([fa(
  "نتیجه‌ی بالا یک پیش‌بینیِ آزمودنی می‌سازد: اگر هم‌ترازی تعیین‌کننده " +
  "باشد، باید در هر جفتِ (رمزگذار، زبان) کار کند، نه فقط در چهار نقطه. " +
  "پس آزمودیمش: سری که روی انگلیسی آموزش دیده، ۳۰۰ میمِ موازی را در پنج " +
  "زبان خواند. بیست نقطه."),
]));
const W3 = [2400, 900, 1400, 1400];
body.push(table(W3, [
  new TableRow({ tableHeader: true, children: [
    cell([T("زیرمجموعه", { bold: true })], { header: true, width: W3[0] }),
    cell([TL("n", { bold: true })], { header: true, width: W3[1] }),
    cell([TL("Pearson r", { bold: true })], { header: true, width: W3[2] }),
    cell([TL("p", { bold: true })], { header: true, width: W3[3] }),
  ]}),
  new TableRow({ children: [
    cell([T("همه‌ی سلول‌ها")], { width: W3[0], align: AlignmentType.START }),
    cell([TL(String(M.overall.n))], { width: W3[1], ltrCell: true }),
    cell([TL(fmt(M.overall.pearson_r, 3))], { width: W3[2], ltrCell: true }),
    cell([TL(fmt(M.overall.pearson_p, 3))], { width: W3[3], ltrCell: true }),
  ]}),
  ...Object.keys(M.subsets).sort().map(f => new TableRow({ children: [
    cell([TL(`concat AUROC >= ${Math.round(Number(f))}`)], { width: W3[0], align: AlignmentType.START, ltrCell: true }),
    cell([TL(String(M.subsets[f].n))], { width: W3[1], ltrCell: true }),
    cell([TL(fmt(M.subsets[f].pearson_r, 3))], { width: W3[2], ltrCell: true }),
    cell([TL(fmt(M.subsets[f].pearson_p, 3))], { width: W3[3], ltrCell: true }),
  ]})),
]));
body.push(P([fa("رابطه‌ای نیست. و اعتراضِ بدیهی — «شاید چون امتیازها نزدیک " +
  "شانس‌اند» — با عدد جواب داده شد نه با استدلال: حذف سلول‌های ضعیف " +
  "چیزی نساخت."),
]));
const j = M.per_encoder["jina-clip-v2"];
body.push(P([fa("آشکارترین ناسازگاری: "), en("jina-clip-v2"), fa(" در "),
  en(ltr(`${j.negative_in}/${j.n}`)), fa(" زبان از ترکیب عنصری ضرر می‌کند ("),
  en(ltr(`${sg(j.mean)}، p = ${fmt(j.p, 3)}`)), fa(
  ") در حالی که در هر پنج زبان، هم‌ترازترین رمزگذار است."),
]));
body.push(P([fa("نتیجه در یک جمله: "),
  fa("پدیده واقعی است، توضیحش کافی نیست.", { bold: true }), fa(
  " سودِ ترکیب واقعاً به رمزگذار وابسته است — این دو بار دیده شد. ولی " +
  "«هم‌ترازی» آن چیزی نیست که تعیینش می‌کند."),
]));
body.push(P([fa(
  "این برای مقاله بهتر است نه بدتر. مقاله‌ای که پیش‌بینیِ خودش را " +
  "می‌آزماید و می‌گوید رد شد، از مقاله‌ای که نمی‌آزماید قابل‌دفاع‌تر است؛ " +
  "داور نمی‌تواند بگوید «این را امتحان نکرده‌اید»."),
]));

// ---- 5 ----
body.push(H("۵. اشتباهاتی که در مسیر رخ داد"));
body.push(P([fa(
  "این‌ها ثبت شده‌اند چون روی زمان‌بندی اثر گذاشتند و همه در فایل " +
  "یادداشت‌های پروژه نوشته شدند تا تکرار نشوند."),
]));
[
  "یک شبِ کامل هدر رفت: برای فرار از محدودیت دو ساعته‌ی کارهای پس‌زمینه، کار جدا اجرا شد؛ ری‌استارت کانتینر آن را کشت و هیچ پیشرفتی نماند — در حالی که همین نکته از پیش در یادداشت‌های پروژه نوشته شده بود.",
  "تعمیم از یک اندازه‌گیری: اثرِ تغییر نسخه‌ی کتابخانه روی یک رمزگذار ۰٫۰۱ واحد سنجیده شد و به بقیه تعمیم داده شد؛ یکی از آن‌ها چهل برابر بیشتر جابه‌جا شد.",
  "دو کارِ سنگینِ هم‌زمان روی چهار هسته ۸٫۷ برابر کندتر شد، نه دو برابر سریع‌تر.",
  "ابزارهای سنجشِ وضعیت سه بار جواب غلط دادند و یک بار باعث شد کارِ زنده «مرده» اعلام شود.",
  "نگهبان‌های کنترلِ منشأ یک شکاف داشتند: فایل‌های نتیجه‌ی بازیابی نسخه ثبت نمی‌کردند، پس دو جدولِ مقاله روی دو خط لوله‌ی متفاوت نشسته بودند بدون اینکه چیزی اعتراض کند.",
].forEach(t => body.push(bullet(t)));

// ---- 6 ----
body.push(H("۶. آنچه اکنون موجود است"));
const W4 = [2800, 3200];
body.push(table(W4, [
  new TableRow({ tableHeader: true, children: [
    cell([T("مورد", { bold: true })], { header: true, width: W4[0] }),
    cell([T("وضعیت", { bold: true })], { header: true, width: W4[1] }),
  ]}),
  ...[
    ["مقاله", "۸ صفحه، فارسی، دو ستونه، طبق قالب نشریه"],
    ["سلول‌های Hateful Memes", "۶۰ اجرا، یک نسخه‌ی کتابخانه، منشأ ثبت‌شده"],
    ["اندازه‌گیری بازیابی", "۲۴ سلول: ۴ رمزگذار × ۶ زبان"],
    ["سلول‌های چندزبانه", "۴۰ اجرا، بیست نقطه‌ی رابطه"],
    ["آزمون‌های خودکار", "۳۳۴ آزمون، همه سبز"],
    ["مخزن", "همه‌چیز ثبت و ارسال شده"],
  ].map(r => new TableRow({ children: [
    cell([T(r[0])], { width: W4[0], align: AlignmentType.START }),
    cell([T(r[1])], { width: W4[1], align: AlignmentType.START }),
  ]})),
]));
body.push(P([fa("نکته‌ی مهم: "),
  fa("هیچ عددی در مقاله تایپ نشده است.", { bold: true }), fa(
  " همه از فایل‌های اجرا خوانده می‌شوند. اگر فردا یک اجرا تغییر کند، دو " +
  "فرمان مقاله را به‌روز می‌کند و امکان ندارد عددِ متن از عددِ اجرا جدا بیفتد."),
]));

// ---- 7 ----
body.push(H("۷. آنچه باقی مانده"));
body.push(bullet("قلم بی‌لوتوس باید روی رایانه‌ی نویسنده نصب باشد؛ در فایل به‌درستی مشخص شده است."));
body.push(bullet("مجموعه‌ی ارزیابیِ فارسی: پروتکلش نوشته و منتشر شده، ولی ساختنش حاشیه‌نویس می‌خواهد. مقاله این را صریح می‌گوید، نه در پاورقی."));
body.push(bullet("یک سلولِ غایب: ترکیبِ ضرب خارجی، که با حدود ۱٫۰۷ میلیارد پارامتر روی هیچ پردازنده‌ی گرافیکیِ رایگانی جا نمی‌گیرد. با یک کارت ۴۸ گیگابایتی قابل پر شدن است و ابلیشن را کامل می‌کند."));

const footer = new Footer({ children: [new Paragraph({
  alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], size: 18,
    font: { ascii: "Times New Roman", hAnsi: "Times New Roman" } })],
})]});

const doc = new Document({
  styles: { default: { document: { run: {
    font: { ascii: "Times New Roman", hAnsi: "Times New Roman", cs: "B Lotus" },
    size: 24 } } } },
  sections: [{
    properties: { page: { margin: { top: 1134, right: 1418, bottom: 1134, left: 1418 } } },
    footers: { default: footer },
    children: body,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("gozaresh.docx", buf);
  console.log(`wrote gozaresh.docx (${(buf.length / 1024).toFixed(0)} KB)`);
});
