const B = require("./build.js");
const F = require("./content_f.js");
const G = require("./content_g.js");
const { P, H, caption, cell, table, T, TL, ltr, COL, fmt, signed, N,
        TableRow, AlignmentType, fa, en, SIZE } = B;

const ENC = ["CLIP ViT-L/14", "CLIP ViT-B/32", "jina-clip-v2", "CLIPfa"];
const au = (e, k) => N.cells[e][k].test.auroc;
const ret = (e, l, k) => N.retrieval[e]?.[l]?.[k];
const star = (p) => (p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : "n.s.");
const cellLine = (e, k) => `${fmt(au(e, k).mean)} ± ${fmt(au(e, k).sd)}`;

const body = [];

// ---------- 3. method ----------
body.push(H("۳- روش", 1));
body.push(P([fa(
  "رمزگذار در تمام آزمایش‌ها منجمد است و تنها سرِ رده‌بند آموزش می‌بیند. " +
  "این انتخاب دو فایده دارد: ویژگی‌ها یک‌بار محاسبه و ذخیره می‌شوند، پس یک " +
  "اجرای آموزش دقیقه‌ای طول می‌کشد و پنج بذر در دسترس است؛ و متغیرِ موردِ " +
  "مطالعه، یعنی خودِ فضای ویژگی، در طول آموزش تغییر نمی‌کند."),
], { indent: { firstLine: 0 } }));

body.push(H("۳-۱ گونه‌های ترکیب", 2));
body.push(P([fa("بردارهای تصویر و متن هرکدام به فضای میانیِ "), en("n"),
  fa("-بُعدی تصویر می‌شوند و سپس ترکیب می‌گردند. الحاق ("), en("concat"),
  fa(") دو بردار را کنار هم می‌گذارد و "), en("2n"),
  fa(" بُعد به سر می‌دهد. ترکیب عنصری ("), en("align"),
  fa(") آن دو را عنصر‌به‌عنصر ضرب می‌کند و "), en("n"),
  fa(" بُعد می‌دهد. تفاوت این دو، کمیتی است که در سراسر مقاله «سودِ ترکیب» " +
  "نامیده می‌شود و همواره به‌صورت تفاضلِ جفتی روی بذرهای مشترک محاسبه شده است."),
], { indent: { firstLine: 0 } }));

body.push(P([fa(
  "اهمیتِ جفتی‌بودن کم نیست. اختلاف‌هایی که این مقاله گزارش می‌کند زیر یک " +
  "واحد "), en("AUROC"), fa(
  " هستند، و نویزِ ناشی از بذر تصادفی بخش عمده‌ی چیزی است که دو میله‌ی " +
  "خطای مستقل نشان می‌دهند. چون هر دو گونه بر یک بذر اجرا شده‌اند، تفاضل " +
  "به‌ازای هر بذر این نویز مشترک را حذف می‌کند. آزمون، "),
  en("t"), fa("ِ جفتیِ دوطرفه است."),
]));

body.push(...G.fusionMath);

body.push(H("۳-۲ هم‌ترازی به‌عنوان کمیتی اندازه‌گیری‌شده", 2));
body.push(P([fa(
  "برای هر رمزگذار، بازیابیِ صفر-شات روی برشی ثابت از "),
  en("Crossmodal-3600"), fa(
  " انجام می‌شود: هر زیرنویس یک پرس‌وجو است و اصابت آن است که تصویر خودش " +
  "در رتبه‌ی نخست بیاید. هیچ رده‌بندی، هیچ برچسبی و هیچ آموزشی در کار " +
  "نیست. سطح شانس "), en(ltr(`${fmt(100 / (ret("CLIP ViT-L/14", "en", "images") || 1000), 2)}٪`)),
  fa(" است و در کنار هر عدد گزارش می‌شود؛ یک فراخوانی بدون سطح شانسِ خود " +
  "بی‌معناست."),
], { indent: { firstLine: 0 } }));

body.push(...F.method);

// ---------- 4. data ----------
body.push(H("۴- داده و پیکربندی آزمایش", 1));
body.push(P([fa("آموزش و ارزیابی روی پیکره‌ی "), en("Hateful Memes Challenge"),
  fa(" ["), en("6"), fa(
  "] انجام می‌شود: ۸۵۰۰ نمونه‌ی آموزش، ۵۰۰ اعتبارسنجی و ۱۰۰۰ آزمون. " +
  "انتخاب مدل بر پایه‌ی "), en("AUROC"), fa(
  " اعتبارسنجی است و بخش آزمون تنها یک‌بار، در پایان، با وزن‌های انتخاب‌شده " +
  "خوانده می‌شود."),
], { indent: { firstLine: 0 } }));

body.push(P([fa("چهار رمزگذار: "), en("CLIP ViT-L/14"), fa(" و "),
  en("CLIP ViT-B/32"), fa(" ["), en("7"), fa("]، که انگلیسی‌محورند؛ "),
  en("jina-clip-v2"), fa(
  "، که چندزبانه است؛ و "), en("CLIPfa"), fa(
  "، که دو بُرجِ جداگانه‌ی فارسی است. دو مورد آخر آینه‌ی یکدیگرند و همین " +
  "است که مجموعه را کنترل‌شده می‌کند: هرکدام در یک زبان قوی و در دیگری در " +
  "سطح شانس است، پس محورِ متغیر «پوششِ زبانی» است نه «کیفیت مدل»."),
]));

body.push(P([fa(
  "هر سلول با پنج بذر (۰ تا ۴) اجرا شده و همه‌ی اعداد میانگین ± انحراف " +
  "معیارِ نمونه‌ای‌اند. تمام ویژگی‌ها زیر یک نسخه‌ی کتابخانه استخراج شده‌اند؛ " +
  "در بخش ۷ نشان می‌دهیم چرا این قید لازم بود."),
]));

body.push(...F.metrics);

// ---------- 5. results ----------
body.push(H("۵- نتایج", 1));
body.push(H("۵-۱ جدول اصلی", 2));
body.push(P([fa("جدول ("), en("1"), fa(") دوازده سلول را نشان می‌دهد."),
], { indent: { firstLine: 0 } }));

body.push(caption("جدول (۱): AUROC آزمون، میانگین ± انحراف معیار روی پنج بذر"));
const W1 = [1500, 1050, 1050, 1000];
body.push(table(W1, [
  new TableRow({ tableHeader: true, children: [
    cell([T("رمزگذار", { bold: true })], { header: true, width: W1[0] }),
    cell([TL("concat", { bold: true })], { header: true, width: W1[1] }),
    cell([TL("align", { bold: true })], { header: true, width: W1[2] }),
    cell([TL("MemeCLIP", { bold: true })], { header: true, width: W1[3] }),
  ]}),
  ...ENC.map((e) => new TableRow({ children: [
    cell([TL(e)], { width: W1[0], align: AlignmentType.START }),
    cell([TL(cellLine(e, "hateclipper/concat"))], { width: W1[1], ltr: true }),
    cell([TL(cellLine(e, "hateclipper/align"))], { width: W1[2], ltr: true }),
    cell([TL(cellLine(e, "memeclip/align"))], { width: W1[3], ltr: true }),
  ]})),
]));

body.push(P([fa(
  "پراکندگیِ میان رمزگذارها بسیار بزرگ‌تر از هر اثر معماری یا ترکیبی است: "),
  en(ltr(`${fmt(Math.abs(N.encoder_gap["CLIP ViT-L/14 - CLIPfa (concat)"].mean))}`)),
  fa(" واحد میان بالاترین و پایین‌ترین ("),
  en(`p < 0.001`), fa(
  "). هر برتریِ معماری که تنها روی یک رمزگذار گزارش شود باید روی رمزگذار " +
  "دیگری بازبینی گردد."),
], { spacing: { before: 140, after: 100 } }));

// ---------- 5.2 alignment ----------
body.push(H("۵-۲ هم‌ترازیِ اندازه‌گیری‌شده", 2));
body.push(P([fa("جدول ("), en("2"), fa(
  ") کمیتی را نشان می‌دهد که هیچ آموزشی آن را نساخته است."),
], { indent: { firstLine: 0 } }));

body.push(caption("جدول (۲): بازیابی صفر-شات روی Crossmodal-3600، R@1 بر حسب درصد"));
const W2 = [1500, 1050, 1050, 1000];
body.push(table(W2, [
  new TableRow({ tableHeader: true, children: [
    cell([T("رمزگذار", { bold: true })], { header: true, width: W2[0] }),
    cell([TL("en t→i", { bold: true })], { header: true, width: W2[1] }),
    cell([TL("en i→t", { bold: true })], { header: true, width: W2[2] }),
    cell([TL("fa t→i", { bold: true })], { header: true, width: W2[3] }),
  ]}),
  ...ENC.map((e) => new TableRow({ children: [
    cell([TL(e)], { width: W2[0], align: AlignmentType.START }),
    cell([TL(fmt(ret(e, "en", "t2i_r1")))], { width: W2[1], ltr: true }),
    cell([TL(fmt(ret(e, "en", "i2t_r1")))], { width: W2[2], ltr: true }),
    cell([TL(fmt(ret(e, "fa", "t2i_r1")))], { width: W2[3], ltr: true }),
  ]})),
  new TableRow({ children: [
    cell([T("سطح شانس")], { width: W2[0], align: AlignmentType.START }),
    cell([TL(fmt(ret("CLIP ViT-L/14", "en", "chance")))], { width: W2[1], ltr: true }),
    cell([TL(fmt(ret("CLIP ViT-L/14", "en", "chance")))], { width: W2[2], ltr: true }),
    cell([TL(fmt(ret("CLIP ViT-L/14", "fa", "chance")))], { width: W2[3], ltr: true }),
  ]}),
]));

body.push(P([fa("دو خانواده‌ی "), en("CLIP"), fa(
  " روی فارسی عملاً در سطح شانس‌اند، و "), en("CLIPfa"), fa(
  " روی انگلیسی همین‌طور. این تقارن است که اجازه می‌دهد اثرِ زبان را از " +
  "اثرِ کیفیتِ مدل جدا کنیم."),
], { spacing: { before: 140, after: 100 } }));

module.exports = { body };
