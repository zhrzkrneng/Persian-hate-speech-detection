const B = require("./build.js");
const { P, H, caption, cell, table, T, TL, ltr, fmt, signed, N,
        TableRow, Paragraph, TextRun, AlignmentType, fa, en, SIZE } = B;

const ben = (e) => N.fusion_benefit[e];
const ret = (e, l, k) => N.retrieval[e]?.[l]?.[k];
const tok = (model, lang) =>
  (N.tokenisation.rows || []).find((r) => r.model === model && r.language === lang);

const E = require("./content_e.js");
const body = [];

// ---------- 5.5 tokenisation ----------
body.push(H("۵-۵ هزینه‌ی توکن‌سازیِ فارسی", 2));
const b32en = tok("openai/clip-vit-base-patch32", "en");
const b32fa = tok("openai/clip-vit-base-patch32", "fa");
const jnen = tok("jinaai/jina-clip-v2", "en");
const jnfa = tok("jinaai/jina-clip-v2", "fa");
if (b32en && b32fa && jnen && jnfa) {
  body.push(P([fa(
    "سنجه‌ی دومی که تنها از توکن‌سازها به دست می‌آید، علتِ این اختلافِ " +
    "هم‌ترازی را نشان می‌دهد. برای زیرنویس‌هایی از همان تصاویر، "),
    en("CLIP"), fa(" به‌طور میانگین "), en(fmt(b32fa.mean_tokens)), fa(
    " توکن برای یک زیرنویسِ فارسی خرج می‌کند و "), en(fmt(b32en.mean_tokens)),
    fa(" برای یک زیرنویسِ انگلیسی — نسبت "),
    en(ltr(`${fmt(b32fa.mean_tokens / b32en.mean_tokens)}×`)), fa(" — در حالی که "),
    en("jina"), fa(" نسبت "),
    en(ltr(`${fmt(jnfa.mean_tokens / jnen.mean_tokens)}×`)), fa(" دارد."),
  ], { indent: { firstLine: 0 } }));

  body.push(P([fa("افزون بر این، پنجره‌ی "), en("CLIP"), fa(
    " ۷۷ توکن است — محدودیتی معماری، نه یک تنظیم — و "),
    en(`${b32fa.over_clip_context_pct}٪`), fa(
    " از زیرنویس‌های فارسی را می‌بُرد، در حالی که برای "), en("jina"),
    fa(" این عدد صفر است. بریدنِ یک‌نهم پرس‌وجوها به‌تنهایی نمی‌تواند " +
    "امتیازی در "), fa("سطح شانس", { bold: true }), fa(
    " بسازد؛ پس این شاهدی بر علت است نه یک متغیرِ مزاحم: همان ناهمخوانیِ " +
    "واژگان هم در شمار توکن و هم در امتیاز بازیابی دیده می‌شود."),
  ]));
}

body.push(...E.afterTok);

body.push(...require("./content_i.js").body);

// ---------- 7. limitations ----------
body.push(H("۷- محدودیت‌ها", 1));
body.push(P([fa("سطح مطلق با عددِ منتشرشده برابر نیست. "), en("Hate-CLIPper"),
  fa(" برای این بخش "), en("85.8"), fa(" گزارش می‌کند و بهترین سلولِ ما "),
  en(fmt(N.cells["CLIP ViT-L/14"]["memeclip/align"].test.auroc.mean)), fa(
  " است. اثرِ کیفی بازتولید می‌شود ولی این خط لوله در عدد مطلق ضعیف‌تر است. " +
  "هر ادعای این مقاله مقایسه‌ای "), fa("درونِ", { italics: true }), fa(
  " همین خط لوله است و هیچ عددی از این مقاله نباید در برابر عددِ مقاله‌ی " +
  "دیگری خوانده شود."),
], { indent: { firstLine: 0 } }));

body.push(P([fa("چهار رمزگذار چهار نقطه است. آزمون‌های جفتی و برهم‌کنش، هرکدام " +
  "روی پنج بذرِ مشترک، معتبرند؛ ولی "), fa("شکلِ", { bold: true }), fa(
  " رابطه از چهار نقطه خوانده شده و اثبات نشده است. ضریب همبستگیِ "),
  en("Pearson"), fa(" روی این چهار نقطه بزرگ به نظر می‌رسد در حالی که " +
  "ضریب رتبه‌ای "), en("Spearman"), fa(
  " کوچک است؛ یعنی خط را یک نقطه‌ی حاشیه‌ای نگه داشته و ترتیبِ بقیه از آن " +
  "پیروی نمی‌کند. این دقیقاً امضای یک رابطه‌ی غیریکنواست و به‌همین دلیل " +
  "هیچ برازشِ خطی‌ای گزارش نمی‌شود."),
]));

body.push(P([fa("سودِ منفیِ "), en("CLIPfa"), fa(
  " در ناحیه‌ای اندازه‌گیری شده که مدل به‌سختی کار می‌کند: "),
  en(fmt(N.cells["CLIPfa"]["hateclipper/concat"].test.auroc.mean)), fa(
  " در برابر سطح شانسِ ۵۰. این اثر واقعی و معنادار است، ولی اثری درون یک " +
  "رژیمِ نزدیک به شکست است و باید همین‌طور توصیف شود."),
]));

body.push(P([fa("ستون اعتبارسنجی برای مقایسه‌ی گونه‌ها به کار نمی‌آید. آن عدد، " +
  "بیشینه‌ی بیست دوره است که انتخابِ مدل برداشته، پس اریبِ رو به بالا دارد " +
  "و تضمینی نیست که این اریب میان گونه‌ها برابر باشد: "), en("align"), fa(" به سر "),
  en("n"), fa(" ورودی می‌دهد و "), en("concat"), fa(" مقدار "), en("2n"),
  fa(". تنها ستونِ آزمون مقایسه می‌کند."),
]));

body.push(P([fa(
  "سرانجام، نسخه‌ی کتابخانه اهمیت دارد و این را با اندازه‌گیری دریافتیم. " +
  "ویژگی‌های متن میان دو نسخه‌ی اصلیِ کتابخانه بیت‌به‌بیت یکسان‌اند ولی " +
  "ویژگی‌های تصویر نیستند، چون پردازشگرِ پیش‌فرضِ تصویر تغییر کرده است. " +
  "اثرش بر "), en("AUROC"), fa(
  " حدود ۰٫۰۱ واحد است، یعنی تقریباً یک‌سی‌ام انحراف معیارِ بذر؛ با این " +
  "حال تمام ویژگی‌های این مقاله زیر یک نسخه استخراج شده‌اند و کُد، کَشی را " +
  "که زیر نسخه‌ی دیگری ساخته شده باشد رد می‌کند."),
]));

// ---------- 8. conclusion ----------
body.push(H("۸- نتیجه‌گیری", 1));
body.push(P([fa(
  "سودِ ترکیبِ چندوجهی خاصیتِ رمزگذار است، نه ظرفیت آن. دو نسخه از یک " +
  "خانواده با فاصله‌ی معنادار در امتیاز مطلق، سودِ ترکیبِ تفکیک‌ناپذیر " +
  "دارند، و رمزگذاری که معنادار ضعیف‌تر است سود بیشتری می‌برد. این روی " +
  "پیکره‌ی انگلیسی برقرار است و روی بیست سلولِ چندزبانه هم، در سطحِ " +
  "رمزگذار، دوباره دیده می‌شود."),
], { indent: { firstLine: 0 } }));

body.push(P([fa(
  "ولی توضیحی که برای این پدیده پیشنهاد شده — ناهم‌ترازیِ فضای ویژگی — " +
  "آزمودنش را تاب نیاورد. روی چهار رمزگذار، سود با هم‌ترازی علامت عوض " +
  "می‌کرد؛ روی بیست جفتِ (رمزگذار، زبان) هیچ رابطه‌ای نیست، و حذف " +
  "سلول‌های نزدیک شانس هم چیزی نمی‌سازد. آشکارترین ناسازگاری این است که "),
  en("jina-clip-v2"), fa(
  " در هر پنج زبان از ترکیب عنصری ضرر می‌کند در حالی که در هر پنج، " +
  "هم‌ترازترین رمزگذار است."),
]));

body.push(P([fa(
  "پس آنچه این مقاله می‌گذارد یک پدیده‌ی تأییدشده و یک توضیحِ ردشده است. " +
  "برای کسی که سامانه‌ای در زبانی کم‌منبع می‌سازد، پیامدِ عملی این است: " +
  "سازوکارِ ترکیب را نمی‌توان از روی هم‌ترازی انتخاب کرد و باید برای " +
  "هر رمزگذار سنجیده شود — ولی انتخابِ خودِ رمزگذار چند برابرِ آن " +
  "اهمیت دارد، و آن را می‌توان بدون هیچ داده‌ی برچسب‌خورده‌ای سنجید."),
]));

body.push(...require("./content_h.js").future);
body.push(...E.repro);

// ---------- acknowledgement (اقرار) ----------
body.push(H("اقرار", 1));
body.push(P([fa("پیکره‌ی "), en("Hateful Memes Challenge"), fa(
  " متعلق به شرکت متا و تحت مجوزِ پذیرشِ کاربر است؛ هیچ بخشی از آن در " +
  "مخزنِ این پژوهش بازتوزیع نشده و اسکریپتِ تهیه‌ی داده تنها نسخه‌ی محلیِ " +
  "کاربر را وارسی می‌کند. زیرنویس‌های "), en("Crossmodal-3600"), fa(
  " از گوگل و تحت "), en("CC BY 4.0"), fa(" است. وزن‌های "), en("jina-clip-v2"),
  fa(" تحت "), en("CC BY-NC 4.0"), fa(
  " منتشر شده‌اند — یعنی پژوهشی و غیرتجاری — و بارگذاری آن‌ها اجرای کدِ " +
  "دریافت‌شده از مخزنِ مدل را لازم دارد. محاسباتِ سنگینِ این کار روی " +
  "پردازنده‌ی مرکزیِ یک نشستِ ابری و پردازنده‌ی گرافیکیِ رایگانِ سامانه‌ی "),
  en("Kaggle"), fa(" انجام شده است؛ هیچ سخت‌افزار اختصاصی‌ای در کار نبوده است."),
], { indent: { firstLine: 0 } }));

// ---------- references ----------
body.push(H("منابع", 1));
const REFS = [
  "G. K. Kumar and K. Nandakumar, “Hate-CLIPper: Multimodal hateful meme classification based on cross-modal interaction of CLIP features,” in Proc. 2nd Workshop on NLP for Positive Impact (WOAH), 2022, pp. 171–183. doi: 10.18653/v1/2022.nlp4pi-1.20",
  "S. Shah, S. Shrestha, S. Poudel, et al., “MemeCLIP: Leveraging CLIP representations for multimodal meme classification,” in Proc. EMNLP, 2024, pp. 17320–17332. doi: 10.18653/v1/2024.emnlp-main.959",
  "A. V. Thapliyal, J. Pont-Tuset, X. Chen, and R. Soricut, “Crossmodal-3600: A massively multilingual multimodal evaluation dataset,” in Proc. EMNLP, 2022, pp. 715–729. doi: 10.18653/v1/2022.emnlp-main.45",
  "J. Mei, J. Chen, W. Lin, B. Byrne, and M. Tomalin, “Improving hateful meme detection through retrieval-guided contrastive learning,” in Proc. ACL, 2024. doi: 10.48550/arXiv.2311.08110",
  "M. D. Bui, K. von der Wense, and A. Lauscher, “Multi3Hate: Multimodal, multilingual, and multicultural hate speech detection with vision-language models,” in Proc. NAACL, 2025. doi: 10.48550/arXiv.2411.03888",
  "D. Kiela, H. Firooz, A. Mohan, et al., “The Hateful Memes Challenge: Detecting hate speech in multimodal memes,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, 2020, pp. 2611–2624. doi: 10.48550/arXiv.2005.04790",
  "A. Radford, J. W. Kim, C. Hallacy, et al., “Learning transferable visual models from natural language supervision,” in Proc. ICML, 2021, pp. 8748–8763. doi: 10.48550/arXiv.2103.00020",
];
REFS.forEach((text, i) => {
  body.push(new Paragraph({
    bidirectional: false, alignment: AlignmentType.LEFT,
    spacing: { after: 70 }, indent: { left: 283, hanging: 283 },
    children: [new TextRun(en(`[${i + 1}] ${text}`, { size: SIZE.refEn + 1 }))],
  }));
});

// ---------- English abstract ----------
body.push(H("Abstract", 1));
body.push(new Paragraph({
  bidirectional: false, alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 100 },
  children: [new TextRun(en(
    "Hateful meme detection requires combining an image with its text, and the " +
    "established architectures do so by multiplying two feature vectors " +
    "element-wise. That mechanism has been reported to give no advantage for a " +
    "language the encoder never saw, attributed to misaligned feature spaces, " +
    "but the attribution had not been tested. We test it. Four frozen encoders " +
    "are trained on the Hateful Memes Challenge over five seeds, and each " +
    "encoder's alignment is measured independently of training by zero-shot " +
    "retrieval on Crossmodal-3600. The fusion benefit changes sign with " +
    "alignment: it harms at chance-level alignment, helps at moderate " +
    "alignment, and does nothing at high alignment. The decisive control is a " +
    "pair of variants of one encoder family that differ significantly in " +
    "absolute score yet have statistically indistinguishable fusion benefits, " +
    "while a significantly weaker encoder gains more than a stronger one, so " +
    "capacity is not what decides. We then test the proposed explanation over " +
    "twenty encoder-language cells, reading 300 parallel memes zero-shot in " +
    "five languages. Alignment does not predict the fusion benefit, and " +
    "dropping the cells near chance does not recover a relationship; what " +
    "generalises is dependence on the encoder, not the alignment axis. The " +
    "phenomenon is real and the usual explanation for it is insufficient. No " +
    "public Persian hateful-meme dataset exists and no Persian number is " +
    "reported.",
    { size: SIZE.body }))],
}));
body.push(new Paragraph({
  bidirectional: false, alignment: AlignmentType.LEFT,
  children: [new TextRun(en(
    "Keywords: hateful memes, multimodal learning, frozen encoders, " +
    "cross-modal alignment, low-resource languages, Persian NLP",
    { size: SIZE.body, bold: true }))],
}));

module.exports = { body };
