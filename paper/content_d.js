const B = require("./build.js");
const { P, H, caption, cell, table, T, TL, ltr, fmt, signed, N,
        TableRow, Paragraph, TextRun, AlignmentType, fa, en, SIZE } = B;

const ben = (e) => N.fusion_benefit[e];
const ret = (e, l, k) => N.retrieval[e]?.[l]?.[k];
const tok = (model, lang) =>
  (N.tokenisation.rows || []).find((r) => r.model === model && r.language === lang);

const E = require("./content_e.js");
// The seed count is data, not prose: the English abstract said "five seeds"
// for a week after every cell had been re-run at fifteen, because the Persian
// abstract was updated by hand and this one was not. It now comes from the
// records, and tests/test_paper_abstracts.py fails if any abstract disagrees.
const SEED_WORDS = { 5: "five", 10: "ten", 15: "fifteen", 20: "twenty" };
const SEED_N = N.fusion_benefit["CLIP ViT-L/14"].n;
const SEED_WORD = SEED_WORDS[SEED_N] || String(SEED_N);

const body = [];

// ---------- 5.5 tokenisation ----------
body.push(H("۵-۵ هزینه‌ی توکن‌سازی فارسی", 2));
const b32en = tok("openai/clip-vit-base-patch32", "en");
const b32fa = tok("openai/clip-vit-base-patch32", "fa");
const jnen = tok("jinaai/jina-clip-v2", "en");
const jnfa = tok("jinaai/jina-clip-v2", "fa");
if (b32en && b32fa && jnen && jnfa) {
  body.push(P([fa(
    "سنجه‌ی دومی که تنها از توکن‌سازها به دست می‌آید، علت این اختلاف " +
    "هم‌ترازی را نشان می‌دهد. برای زیرنویس‌هایی از همان تصاویر، "),
    en("CLIP"), fa(" به‌طور میانگین "), en(fmt(b32fa.mean_tokens)), fa(
    " توکن برای یک زیرنویس فارسی خرج می‌کند و "), en(fmt(b32en.mean_tokens)),
    fa(" برای یک زیرنویس انگلیسی — نسبت "),
    en(ltr(`${fmt(b32fa.mean_tokens / b32en.mean_tokens)}×`)), fa(" — در حالی که "),
    en("jina"), fa(" نسبت "),
    en(ltr(`${fmt(jnfa.mean_tokens / jnen.mean_tokens)}×`)), fa(" دارد."),
  ], { indent: { firstLine: 0 } }));

  body.push(P([fa("افزون بر این، پنجره‌ی "), en("CLIP"), fa(
    " ۷۷ توکن است — محدودیتی معماری، نه یک تنظیم — و "),
    en(`${b32fa.over_clip_context_pct}٪`), fa(
    " از زیرنویس‌های فارسی را قطع می‌کند، در حالی که برای "), en("jina"),
    fa(" این عدد صفر است. بریدن یک‌نهم پرس‌وجوها به‌تنهایی نمی‌تواند " +
    "امتیازی در "), fa("سطح شانس", { bold: true }), fa(
    " بسازد؛ پس این شاهدی بر علت است نه یک متغیر مزاحم: همان ناهمخوانی " +
    "واژگان هم در شمار توکن و هم در امتیاز بازیابی دیده می‌شود."),
  ]));
}

body.push(...E.afterTok);

body.push(...require("./content_i.js").body);

// ---------- 7. limitations ----------
body.push(H("۷- محدودیت‌ها", 1));
body.push(P([fa("سطح مطلق پایین‌تر از عدد منتشرشده است، و مقایسه دقت " +
  "می‌خواهد. چکیده‌ی "), en("Hate-CLIPper"), fa(" عدد "), en("85.8"), fa(
  " را می‌دهد، ولی آن سطر مال "), en("align"), fa(" با "),
  fa("سه", { bold: true }), fa(" لایه‌ی پیش‌خروجی است؛ پیکربندی این مقاله " +
  "یک لایه دارد و سطر هم‌پیکربندی‌اش "), en("85.46"), fa(
  " است. هر ۹ فراپارامتر جدول ۳ آن مقاله — اندازه‌ی تصویر، نسخه‌ی "),
  en("CLIP"), fa(
  "، پهنای تصویر و پیش‌خروجی، بهینه‌ساز، دوره‌ها، اندازه‌ی دسته، نرخ " +
  "یادگیری، واپاشی وزن و بریدن گرادیان — با کانفیگ‌های اینجا یکی است، پس " +
  "اختلاف از بودجه یا پهنا یا بهینه‌ساز نمی‌آید."),
], { indent: { firstLine: 0 } }));

body.push(P([fa("در برابر سطر هم‌پیکربندی، تصویر آمیخته است نه یکدست " +
  "بدتر: الحاق اینجا "),
  en(ltr(fmt(N.cells["CLIP ViT-L/14"]["hateclipper/concat"].test.auroc.mean))),
  fa(" است در برابر "), en("79.87"), fa(" آن‌ها — یعنی "),
  fa("بالاتر", { bold: true }), fa(". آنچه فرق دارد اندازه‌ی سود ترکیب " +
  "است: "), en("+5.59"), fa(" در جدول آن‌ها و "),
  en(ltr(signed(N.fusion_benefit["CLIP ViT-L/14"].mean))), fa(
  " اینجا. همان کمیتی که موضوع این مقاله است، پس اختلاف گزارش می‌شود نه " +
  "توجیه، و هیچ ادعایی اینجا به برابری با سطح مطلق آن‌ها متکی نیست."),
]));

body.push(P([fa(
  "دو تفاوت پروتکل هم در خواندن این مقایسه اثر دارد. جدول آن‌ها نه تعداد " +
  "بذر می‌دهد نه انحراف معیار، پس هر سلول تک‌اجرا به نظر می‌رسد، در حالی " +
  "که اختلاف‌های موضوع این مقاله زیر یک واحدند. و در همان جدول، "),
  en("dev-seen"), fa(" گونه‌ها را به ترتیبی که "), en("test-seen"), fa(
  " می‌چیند نمی‌چیند: بهترین سلول "), en("dev"), fa(" ترکیب خارجی با "),
  en("82.62"), fa(" است که روی آزمون "), en("85.12"), fa(
  " می‌گیرد، در حالی که بهترین سلول آزمون روی "), en("dev"), fa(
  " چهارم است. انتخاب روی "), en("dev"), fa(" و خواندن یک‌بار آزمون — " +
  "پروتکل این مقاله — از آن جدول "), en("85.12"), fa(
  " بیرون می‌دهد. این گزاره‌ای درباره‌ی آن جدول است؛ آن مقاله نمی‌گوید " +
  "اعدادش چگونه انتخاب شده‌اند."),
]));

body.push(P([fa("چهار رمزگذار چهار نقطه است. آزمون‌های جفتی و برهم‌کنش، هرکدام " +
  "روی پانزده بذر مشترک، معتبرند؛ ولی "), fa("شکل", { bold: true }), fa(
  " رابطه از چهار نقطه خوانده شده و اثبات نشده است. ضریب همبستگی "),
  en("Pearson"), fa(" روی این چهار نقطه بزرگ به نظر می‌رسد در حالی که " +
  "ضریب رتبه‌ای "), en("Spearman"), fa(
  " کوچک است؛ یعنی خط را یک نقطه‌ی حاشیه‌ای نگه داشته و ترتیب بقیه از آن " +
  "پیروی نمی‌کند. این دقیقا امضای یک رابطه‌ی غیریکنواست و به‌همین دلیل " +
  "هیچ برازش خطی‌ای گزارش نمی‌شود."),
]));

body.push(P([fa("سود منفی "), en("CLIPfa"), fa(
  " در ناحیه‌ای اندازه‌گیری شده که مدل به‌سختی کار می‌کند: "),
  en(ltr(fmt(N.cells["CLIPfa"]["hateclipper/concat"].test.auroc.mean))), fa(
  " در برابر سطح شانس ۵۰. با این حال اثر در پانزده بذر پایدار است ("),
  en(ltr(`${signed(N.fusion_benefit["CLIPfa"].mean)}، p < 0.001`)), fa(
  ")، پس نویز بهینه‌ساز روی نمونه‌ی کوچک نیست. ولی همچنان اثری درون یک " +
  "رژیم نزدیک به شکست است و باید همین‌طور توصیف شود، نه به‌عنوان خاصیتی " +
  "عمومی رمزگذارهای کم‌هم‌تراز."),
]));

body.push(P([fa("ترکیب "), en("cross"), fa(
  " در ابلیشن غایب است: در پهنای ۲۵۶ نخستین لایه‌ی رده‌بند ۶۷٫۶ میلیون " +
  "پارامتر دارد و یک بذر ساعت‌ها روی چهار هسته می‌گیرد؛ در پهنای ۱۰۲۴ " +
  "حدود ۱٫۰۷ میلیارد می‌شود و روی هیچ پردازنده‌ی گرافیکی رایگانی جا " +
  "نمی‌گیرد. هزینه‌ی این غیبت کمتر از آن است که به نظر می‌رسد، و جدول "),
  en("Hate-CLIPper"), fa(" خودش می‌گوید چرا: سلول "), en("cross"), fa(
  " آن‌ها با "), en("1.1B"), fa(" پارامتر به "), en("85.12"), fa(
  " می‌رسد، در برابر "), en("85.46"), fa(" برای "), en("align"), fa(" با "),
  en("2.9M"), fa(" — یعنی "), fa("پایین‌تر", { bold: true }), fa(
  " از گونه‌ی ارزان، با ۳۸۰ برابر پارامتر. سلول غایب یک هزینه‌ی غایب است، " +
  "نه یک برتری غایب، و هیچ ادعایی اینجا به آن متکی نیست."),
]));

body.push(P([fa("ستون اعتبارسنجی برای مقایسه‌ی گونه‌ها به کار نمی‌آید. آن عدد، " +
  "بیشینه‌ی بیست دوره است که انتخاب مدل برداشته، پس اریب رو به بالا دارد " +
  "و تضمینی نیست که این اریب میان گونه‌ها برابر باشد: "), en("align"), fa(" به سر "),
  en("n"), fa(" ورودی می‌دهد و "), en("concat"), fa(" مقدار "), en("2n"),
  fa(". تنها ستون آزمون مقایسه می‌کند."),
]));

body.push(P([fa(
  "سرانجام، نسخه‌ی کتابخانه اهمیت دارد و این را با اندازه‌گیری دریافتیم. " +
  "ویژگی‌های متن میان دو نسخه‌ی اصلی کتابخانه بیت‌به‌بیت یکسان‌اند ولی " +
  "ویژگی‌های تصویر نیستند، چون پردازشگر پیش‌فرض تصویر تغییر کرده است. " +
  "اثرش بر "), en("AUROC"), fa(
  " حدود ۰٫۰۱ واحد است، یعنی تقریبا یک‌سی‌ام انحراف معیار بذر؛ با این " +
  "حال تمام ویژگی‌های این مقاله زیر یک نسخه استخراج شده‌اند و کد، حافظه‌ای را " +
  "که زیر نسخه‌ی دیگری ساخته شده باشد رد می‌کند."),
]));

// ---------- 8. conclusion ----------
body.push(H("۸- نتیجه‌گیری", 1));
body.push(P([fa(
  "اگر بخواهیم کل این پژوهش را در یک جمله بگوییم: سود ترکیب چندوجهی " +
  "به رمزگذار بستگی دارد و نه به ظرفیت آن. شاهد اصلی‌مان آن دو نسخه از " +
  "یک خانواده است که امتیاز مطلقشان به‌روشنی با هم فرق دارد، ولی سود " +
  "ترکیبشان از نظر آماری قابل تفکیک نیست؛ و در کنارش رمزگذاری که " +
  "به‌روشنی ضعیف‌تر است و با این حال سود بیشتری می‌برد. همین الگو هم " +
  "روی پیکره‌ی انگلیسی درست درمی‌آید و هم روی بیست سلول چندزبانه " +
  "دوباره دیده می‌شود."),
], { indent: { firstLine: 0 } }));

body.push(P([fa(
  "ولی توضیحی که برای این پدیده داده شده — ناهم‌ترازی فضای ویژگی — " +
  "از آزمایش سربلند بیرون نیامد. روی چهار رمزگذار، سود با هم‌ترازی " +
  "علامت عوض می‌کرد. روی بیست جفت رمزگذار-زبان هیچ رابطه‌ای نیست، و " +
  "کنار گذاشتن سلول‌های نزدیک شانس هم چیزی درست نمی‌کند. روشن‌ترین " +
  "نمونه‌ی نقض این است که "),
  en("jina-clip-v2"), fa(
  " در هر پنج زبان از ترکیب عنصری ضرر می‌کند در حالی که در هر پنج، " +
  "هم‌ترازترین رمزگذار است."),
]));

body.push(P([fa(
  "پس حاصل این مقاله یک پدیده‌ی تأییدشده است و یک توضیح ردشده. " +
  "می‌دانیم که نتیجه‌ی دوم آن چیزی نیست که معمولا کسی امیدوار است " +
  "گزارش کند، ولی به نظر ما ارزشش از نتیجه‌ی اول کمتر نیست: یک توضیح " +
  "که همه تکرارش می‌کنند و کسی نیازموده، تا وقتی آزموده نشود دانش " +
  "حساب نمی‌شود."),
]));

body.push(P([fa(
  "برای کسی که دارد سامانه‌ای در یک زبان کم‌منبع می‌سازد، چند توصیه‌ی " +
  "عملی از این کار بیرون می‌آید. اول اینکه روش ترکیب را از روی عدد " +
  "هم‌ترازی انتخاب نکنید؛ ما نشان دادیم این کار جواب نمی‌دهد. به‌جایش " +
  "هر دو گونه را روی همان رمزگذاری که قرار است استفاده شود اجرا کنید " +
  "و تفاضلشان را ببینید. چون رمزگذار منجمد است، این آزمایش ارزان است " +
  "و چند ساعت بیشتر وقت نمی‌گیرد. دوم اینکه اگر وقت و بودجه‌تان محدود " +
  "است، آن را صرف انتخاب رمزگذار کنید نه تنظیم روش ترکیب: در جدول ۱ " +
  "فاصله‌ی میان رمزگذارها بیست واحد است و فاصله‌ی میان گونه‌های ترکیب " +
  "کمتر از دو واحد. سوم اینکه کیفیت رمزگذار را می‌شود بدون هیچ داده‌ی " +
  "برچسب‌خورده‌ای سنجید، و برای زبانی که داده‌ی برچسب‌خورده ندارد این " +
  "تنها راه موجود است."),
]));

body.push(...require("./content_h.js").future);
body.push(...E.repro);

// ---------- سپاسگزاری ----------
// The journal's author guidelines list سپاسگزاری among the required sections.
// It is kept separate from اقرار below, which states licences and compute and
// is not thanks. Nothing here claims funding or support that did not exist.
body.push(H("سپاسگزاری", 1));
body.push(P([fa(
  "این پژوهش از هیچ حمایت مالی سازمانی برخوردار نبوده است. محاسبات روی " +
  "منابع رایگان عمومی انجام شده و هیچ سخت‌افزار اختصاصی‌ای در کار نبوده " +
  "است. از داوران ناشناس نشریه برای وقتی که صرف خواندن این مقاله " +
  "می‌کنند سپاسگزاریم."),
], { indent: { firstLine: 0 } }));

// ---------- acknowledgement (اقرار) ----------
body.push(H("اقرار", 1));
body.push(P([fa("پیکره‌ی "), en("Hateful Memes Challenge"), fa(
  " متعلق به شرکت متا و تحت مجوز پذیرش کاربر است؛ هیچ بخشی از آن در " +
  "مخزن این پژوهش بازتوزیع نشده و اسکریپت تهیه‌ی داده تنها نسخه‌ی محلی " +
  "کاربر را وارسی می‌کند. زیرنویس‌های "), en("Crossmodal-3600"), fa(
  " از گوگل و تحت "), en("CC BY 4.0"), fa(" است. وزن‌های "), en("jina-clip-v2"),
  fa(" تحت "), en("CC BY-NC 4.0"), fa(
  " منتشر شده‌اند — یعنی پژوهشی و غیرتجاری — و بارگذاری آن‌ها اجرای کد " +
  "دریافت‌شده از مخزن مدل را لازم دارد. محاسبات سنگین این کار روی " +
  "پردازنده‌ی مرکزی یک نشست ابری و پردازنده‌ی گرافیکی رایگان سامانه‌ی "),
  en("Kaggle"), fa(" انجام شده است؛ هیچ سخت‌افزار اختصاصی‌ای در کار نبوده است."),
], { indent: { firstLine: 0 } }));

// ---------- references ----------
body.push(H("منابع", 1));
const REFS = [
  "G. K. Kumar and K. Nandakumar, “Hate-CLIPper: Multimodal hateful meme classification based on cross-modal interaction of CLIP features,” in Proc. 2nd Workshop on NLP for Positive Impact (NLP4PI), Abu Dhabi, United Arab Emirates, Dec. 2022, pp. 171–183. doi: 10.18653/v1/2022.nlp4pi-1.20",
  "S. B. Shah, S. Shiwakoti, M. Chaudhary, and H. Wang, “MemeCLIP: Leveraging CLIP representations for multimodal meme classification,” in Proc. EMNLP, Miami, FL, USA, Nov. 2024, pp. 17320–17332. doi: 10.18653/v1/2024.emnlp-main.959",
  "A. V. Thapliyal, J. Pont-Tuset, X. Chen, and R. Soricut, “Crossmodal-3600: A massively multilingual multimodal evaluation dataset,” in Proc. EMNLP, Abu Dhabi, United Arab Emirates, Dec. 2022, pp. 715–729. doi: 10.18653/v1/2022.emnlp-main.45",
  "J. Mei, J. Chen, W. Lin, B. Byrne, and M. Tomalin, “Improving hateful meme detection through retrieval-guided contrastive learning,” in Proc. ACL (Vol. 1: Long Papers), Bangkok, Thailand, Aug. 2024, pp. 5333–5347. doi: 10.18653/v1/2024.acl-long.291",
  "M. D. Bui, K. von der Wense, and A. Lauscher, “Multi3Hate: Multimodal, multilingual, and multicultural hate speech detection with vision-language models,” in Proc. NAACL-HLT (Vol. 1: Long Papers), Albuquerque, NM, USA, Apr. 2025, pp. 9714–9731. doi: 10.18653/v1/2025.naacl-long.490",
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
    `are trained on the Hateful Memes Challenge over ${SEED_WORD} seeds, and each ` +
    "encoder's alignment is measured independently of training by zero-shot " +
    "retrieval on Crossmodal-3600. The fusion benefit changes sign with " +
    "alignment: it harms at chance-level alignment, helps at moderate " +
    "alignment, and does nothing at high alignment. The decisive control is a " +
    "pair of variants of one encoder family that differ significantly in " +
    "absolute score yet have statistically indistinguishable fusion benefits, " +
    "while a significantly weaker encoder gains more than a stronger one, so " +
    "capacity is not what decides. We then test the proposed explanation over " +
    "twenty encoder-language cells, reading 300 parallel memes zero-shot in " +
    "five languages. Alignment does not predict the fusion benefit: neither " +
    "dropping the cells near chance, nor correcting for measurement error, " +
    "nor fitting each language separately recovers a relationship. The same " +
    "dependence appears in the architecture: MemeCLIP's three additions help " +
    "on one encoder and hurt on another, both significantly, so the sign " +
    "flips. What generalises is dependence on the encoder, not the alignment " +
    "axis -- the phenomenon is real and the usual explanation for it is " +
    "insufficient. No public Persian hateful-meme dataset exists and no " +
    "Persian number is reported.",
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
