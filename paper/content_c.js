const B = require("./build.js");
const { P, H, caption, cell, table, T, TL, ltr, fmt, signed, N, fs,
        TableRow, Paragraph, TextRun, ImageRun, AlignmentType, fa, en, SIZE } = B;

const ENC = ["CLIPfa", "CLIP ViT-B/32", "CLIP ViT-L/14", "jina-clip-v2"];
const ben = (e) => N.fusion_benefit[e];
const star = (p) => (p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : "n.s.");
const ret = (e, l, k) => N.retrieval[e]?.[l]?.[k];
const inter = (k) => N.interaction[k];
const gap = (k) => N.encoder_gap[k];

const body = [];

// ---------- 5.3 the central result ----------
body.push(H("۵-۳ سود ترکیب علامت عوض می‌کند", 2));
body.push(P([fa("جدول ("), en("3"), fa(
  ") سود ترکیب را برای هر رمزگذار نشان می‌دهد، مرتب‌شده بر حسب هم‌ترازی " +
  "انگلیسی. وقتی این جدول را برای اولین بار دیدیم، چیزی که بیش از همه " +
  "جلب توجه می‌کرد اندازه‌ی اعداد نبود، علامتشان بود. انتظار داشتیم " +
  "ضرب عنصری روی بعضی رمزگذارها بیشتر و روی بعضی کمتر کمک کند؛ اینکه " +
  "روی دو تا از چهار رمزگذار "), fa("ضرر", { bold: true }), fa(
  " بزند چیزی بود که در هیچ‌کدام از مقاله‌های مرجع ندیده بودیم."),
], { indent: { firstLine: 0 } }));

body.push(caption("جدول (۳): سود ترکیب (align منهای concat)، تفاضل جفتی روی پانزده بذر"));
const W3 = [1150, 700, 700, 1100, 650, 700];
body.push(table(W3, [
  new TableRow({ tableHeader: true, children: [
    cell([T("رمزگذار", { bold: true })], { header: true, width: W3[0] }),
    cell([TL("en R@1", { bold: true })], { header: true, width: W3[1] }),
    cell([T("سود", { bold: true })], { header: true, width: W3[2] }),
    cell([T("بازه ۹۵٪", { bold: true })], { header: true, width: W3[3] }),
    cell([TL("d", { bold: true })], { header: true, width: W3[4] }),
    cell([TL("p", { bold: true })], { header: true, width: W3[5] }),
  ]}),
  ...ENC.map((e) => new TableRow({ children: [
    cell([TL(e)], { width: W3[0], align: AlignmentType.START }),
    cell([TL(fmt(ret(e, "en", "t2i_r1")))], { width: W3[1], ltr: true }),
    cell([TL(signed(ben(e).mean), { bold: ben(e).p < 0.05 })], { width: W3[2], ltr: true }),
    cell([TL(`[${signed(ben(e).ci_low)}, ${signed(ben(e).ci_high)}]`)],
      { width: W3[3], ltr: true }),
    cell([TL(signed(ben(e).d))], { width: W3[4], ltr: true }),
    cell([TL(`${fmt(ben(e).p, 3)} ${star(ben(e).p)}`)], { width: W3[5], ltr: true }),
  ]})),
]));

body.push(P([fa(
  "ستون "), en("d"), fa(
  " اندازه‌ی اثر کوهن است: میانگین تفاضل بر حسب انحراف معیار خودش. " +
  "مقدار "), en("p"), fa(
  " می‌گوید آیا اثر از صفر قابل تشخیص است، که با بذر کافی سرانجام همیشه " +
  "می‌شود؛ "), en("d"), fa(
  " می‌گوید نسبت به نویز چقدر بزرگ است، که با بذر بیشتر تغییر نمی‌کند. " +
  "هر چهار اثر این جدول بزرگ‌اند (قرارداد: ۰٫۸ به بالا بزرگ)."),
], { spacing: { before: 140 } }));

body.push(P([fa(
  "دو اثر معنادار در دو جهت مخالف. در هم‌ترازی سطح شانس، ضرب عنصری " +
  "رده‌بند را "), fa("بدتر", { bold: true }), fa(" می‌کند ("),
  en(ltr(`${signed(ben("CLIPfa").mean)}، p = ${fmt(ben("CLIPfa").p, 3)}`)), fa(
  ")؛ در هم‌ترازی متوسط کمک می‌کند ("),
  en(ltr(`${signed(ben("CLIP ViT-B/32").mean)}، p = ${fmt(ben("CLIP ViT-B/32").p, 3)}`)),
  fa(")؛ و در بالاترین هم‌ترازی بی‌اثر است. رابطه یکنوا نیست."),
], { spacing: { before: 140 } }));

body.push(P([fa(
  "سازوکارش خوانا است. ضرب عنصری فرض می‌کند دو فضا مؤلفه‌به‌مؤلفه با هم " +
  "متناظرند. آنجا که چنین است، حاصل‌ضرب یک برهم‌کنش آگاهی‌بخش است. آنجا که " +
  "دو فضا از پیش به‌خوبی بر هم منطبق‌اند، حاصل‌ضرب چیزی به آنچه سر " +
  "رده‌بند از الحاق یاد می‌گیرد اضافه نمی‌کند. و آنجا که هیچ تناظری " +
  "نیست، " +
  "حاصل‌ضرب نویز است و اطلاعاتی را از میان می‌برد که الحاق حفظ می‌کرد. " +
  "مشاهده‌ی تامیلی ["), en("1"), fa("] همین حالت سوم است، که اینجا " +
  "به‌صورت کنترل‌شده و با هم‌ترازی اندازه‌گیری‌شده بازتولید شده است."),
]));

// ---------- 5.4 the capacity control ----------
body.push(H("۵-۴ کنترل ظرفیت", 2));
body.push(P([fa(
  "تا اینجا فقط نشان داده‌ایم که سود ترکیب از رمزگذاری به رمزگذار دیگر " +
  "فرق می‌کند. ولی این به‌تنهایی اثبات نمی‌کند که "),
  fa("هم‌ترازی", { bold: true }), fa(
  " عامل این تفاوت است. یک توضیح رقیب و کاملا طبیعی وجود دارد: شاید " +
  "رمزگذار قوی‌تر فقط به این دلیل سود بیشتری می‌برد که قوی‌تر است. " +
  "برای اینکه این دو را از هم جدا کنیم به یک آزمایش کنترلی نیاز " +
  "داشتیم، نه به یک همبستگی دیگر."),
], { indent: { firstLine: 0 } }));

body.push(P([fa(
  "خوشبختانه مجموعه‌ی رمزگذارها خودش چنین کنترلی در اختیارمان می‌گذاشت. " +
  "دو نسخه‌ی "), en("CLIP"), fa(
  " از یک خانواده‌اند و روی یک داده آموزش دیده‌اند، و هم‌ترازی‌شان هم " +
  "تنها "), en(ltr(fmt(Math.abs(ret("CLIP ViT-L/14", "en", "t2i_r1") - ret("CLIP ViT-B/32", "en", "t2i_r1"))))),
  fa(" واحد "), en("R@1"), fa(
  " با هم فرق دارد. ولی ظرفیتشان به‌روشنی فرق دارد و همین باعث می‌شود " +
  "امتیاز مطلقشان فاصله‌ی زیادی داشته باشد. اگر ظرفیت عامل تعیین‌کننده " +
  "بود، باید سودشان هم به همان نسبت فرق می‌کرد. اگر هم‌ترازی عامل بود، " +
  "باید سودشان نزدیک هم می‌ماند. این دو پیش‌بینی با هم ناسازگارند، و " +
  "همین است که آزمایش را قاطع می‌کند."),
]));

body.push(P([fa("جدول ("), en("4"),
  fa(") همین را می‌پرسد: آیا سود ترکیب میان دو رمزگذار تفاوت دارد؟ " +
  "کمیتی که گزارش می‌شود تفاضل دو تفاضل جفتی است. یعنی اول برای هر " +
  "رمزگذار سود را روی هر بذر حساب می‌کنیم، و بعد این سودها را میان دو " +
  "رمزگذار و باز هم روی بذرهای مشترک از هم کم می‌کنیم. این کار نویز " +
  "مشترک بذرها را دو بار حذف می‌کند و برآورد را بسیار دقیق‌تر از دو " +
  "میانگین مستقل می‌سازد."),
]));

body.push(caption("جدول (۴): برهم‌کنش — آیا سود ترکیب به رمزگذار وابسته است؟"));
const W4 = [2450, 800, 700, 650];
const pairs = Object.keys(N.interaction);
body.push(table(W4, [
  new TableRow({ tableHeader: true, children: [
    cell([T("زوج رمزگذار", { bold: true })], { header: true, width: W4[0] }),
    cell([T("اختلاف", { bold: true })], { header: true, width: W4[1] }),
    cell([TL("sd", { bold: true })], { header: true, width: W4[2] }),
    cell([TL("p", { bold: true })], { header: true, width: W4[3] }),
  ]}),
  ...pairs.map((k) => {
    const isControl = k === "CLIP ViT-L/14 - CLIP ViT-B/32";
    return new TableRow({ children: [
      cell([TL(k.replace(" - ", " − "), { bold: isControl })],
        { width: W4[0], align: AlignmentType.START }),
      cell([TL(signed(inter(k).mean), { bold: isControl })], { width: W4[1], ltr: true }),
      cell([TL(fmt(inter(k).sd))], { width: W4[2], ltr: true }),
      cell([TL(`${fmt(inter(k).p, 3)} ${star(inter(k).p)}`, { bold: isControl })],
        { width: W4[3], ltr: true }),
    ]});
  }),
]));

body.push(P([fa("سطر نخست تعیین‌کننده است. دو نسخه‌ی "), en("CLIP"), fa(
  " در "), en("AUROC"), fa(" مطلق "),
  en(ltr(`${fmt(Math.abs(gap("CLIP ViT-L/14 - CLIP ViT-B/32 (concat)").mean))}`)),
  fa(" واحد فاصله دارند و هم‌ترازی‌شان تنها "),
  en(ltr(`${fmt(Math.abs((ret("CLIP ViT-L/14", "en", "t2i_r1") || 0) - (ret("CLIP ViT-B/32", "en", "t2i_r1") || 0)))}`)),
  fa(" واحد، و سود ترکیبشان از نظر آماری تفکیک‌ناپذیر است ("),
  en(ltr(`p = ${fmt(inter("CLIP ViT-L/14 - CLIP ViT-B/32").p, 3)}`)), fa(
  "). در حالی که هر دو با دو رمزگذار دیگر تفاوت دارند. هم‌ترازی یکسان، " +
  "سود یکسان، ظرفیت متفاوت."),
], { spacing: { before: 140 } }));

body.push(P([fa("و توضیح رقیب — «مدل قوی‌تر بیشتر سود می‌برد» — با یک عدد " +
  "معنادار رد می‌شود، نه با استدلال: "), en("CLIP ViT-B/32"), fa(
  " زیر الحاق به‌طور معنادار از "), en("jina-clip-v2"), fa(" ضعیف‌تر است ("),
  en(ltr(`${signed(gap("CLIP ViT-B/32 - jina-clip-v2 (concat)").mean)}، p < 0.001`)),
  fa(") و با این حال سود ترکیب بزرگ‌تر و معناداری دارد، جایی که "),
  en("jina"), fa(" هیچ سودی ندارد."),
]));

// ---------- figure ----------
const figurePath = "alignment_benefit.png";
if (fs.existsSync(figurePath)) {
  body.push(new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 160, after: 40 },
    children: [new ImageRun({
      type: "png", data: fs.readFileSync(figurePath),
      transformation: { width: 300, height: 190 },
    })],
  }));
  body.push(caption(
    "شکل (۱): هم‌ترازی اندازه‌گیری‌شده‌ی رمزگذار (محور افقی، R@1 بازیابی " +
    "متن به تصویر روی Crossmodal-3600 انگلیسی، بر حسب درصد) در برابر سود " +
    "ترکیب (محور عمودی، اختلاف AUROC آزمون میان align و concat). " +
    "میله‌ها خطای معیار میانگین روی پانزده بذرند. خط‌چین افقی «بدون سود» و " +
    "نقطه‌چین عمودی سطح شانس را نشان می‌دهد."));
}

module.exports = { body };
