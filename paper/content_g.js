// Formal definitions, as numbered equations. The template asks for equations
// to carry a number in parentheses, right-aligned, with every symbol defined.
const B = require("./build.js");
const { P, H, caption, cell, table, T, TL, fmt, N,
        TableRow, Paragraph, TextRun, AlignmentType, WidthType, fa, en, SIZE } = B;
const { BorderStyle, Table, TableCell } = require("docx");

// An equation row: the expression centred, its number at the right margin.
// Built as a borderless table because that is how the journal's own template
// lays one out.
function equation(parts, number) {
  const W = [700, 3900];
  return new Table({
    width: { size: 4600, type: WidthType.DXA },
    columnWidths: W,
    visuallyRightToLeft: true,
    borders: {
      top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.NONE },
      insideVertical: { style: BorderStyle.NONE },
    },
    rows: [new TableRow({ children: [
      new TableCell({
        width: { size: W[0], type: WidthType.DXA },
        margins: { top: 60, bottom: 60 },
        children: [new Paragraph({
          bidirectional: true, alignment: AlignmentType.START,
          spacing: { after: 0 },
          children: [new TextRun(en(`(${number})`))],
        })],
      }),
      new TableCell({
        width: { size: W[1], type: WidthType.DXA },
        margins: { top: 60, bottom: 60 },
        children: [new Paragraph({
          bidirectional: false, alignment: AlignmentType.CENTER,
          spacing: { after: 0 },
          children: parts.map((p) => new TextRun(p)),
        })],
      }),
    ]})],
  });
}

const E = (t, o = {}) => en(t, { size: SIZE.body, ...o });
const I = (t) => en(t, { size: SIZE.body, italics: true });

const fusionMath = [];
fusionMath.push(P([fa(
  "به‌صورت صوری، فرض کنید "), I("v"), fa(" و "), I("t"), fa(
  " به‌ترتیب بردارهای ویژگیِ تصویر و متن باشند که رمزگذارِ منجمد تولید " +
  "کرده است. هر دو با یک تبدیل خطی به فضای میانیِ "), I("n"), fa(
  "-بُعدی برده می‌شوند:"),
]));
fusionMath.push(equation([I("p"), E("ᵥ = "), I("W"), E("ᵥ "), I("v"),
                          E(",     "), I("p"), E("ₜ = "), I("W"), E("ₜ "), I("t")], 1));
fusionMath.push(P([fa("که در آن "), I("W"), E("ᵥ"), fa(" و "), I("W"), E("ₜ"),
  fa(" ماتریس‌های یادگرفته‌شده‌ی تصویرند. دو گونه‌ی ترکیب از همین دو بردار " +
  "ساخته می‌شوند. الحاق، آن‌ها را پشت سر هم می‌گذارد:"),
], { indent: { firstLine: 0 } }));
fusionMath.push(equation([I("f"), E("_concat = ["), I("p"), E("ᵥ ; "),
                          I("p"), E("ₜ] ∈ ℝ"), E("²ⁿ")], 2));
fusionMath.push(P([fa("و ترکیب عنصری، آن‌ها را درایه‌به‌درایه ضرب می‌کند:"),
], { indent: { firstLine: 0 } }));
fusionMath.push(equation([I("f"), E("_align = "), I("p"), E("ᵥ ⊙ "),
                          I("p"), E("ₜ ∈ ℝ"), E("ⁿ")], 3));
fusionMath.push(P([fa("که در آن "), E("⊙"), fa(
  " ضربِ هادامارد است. کمیتی که این مقاله «سودِ ترکیب» می‌نامد، تفاضلِ " +
  "امتیازِ آزمونِ این دو است، به‌ازای هر بذر:"),
], { indent: { firstLine: 0 } }));
fusionMath.push(equation([E("Δ"), I("s"), E(" = AUROC("), I("f"),
                          E("_align) − AUROC("), I("f"), E("_concat)")], 4));
fusionMath.push(P([fa(
  "اهمیتِ صورتِ معادله‌ی ("), en("3"), fa(
  ") همین‌جاست: درایه‌ی "), I("i"), fa("ـُمِ خروجی تنها به درایه‌ی "), I("i"),
  fa("ـُمِ هر دو بردار وابسته است. پس این سازوکار فرض می‌کند که بُعدِ "),
  I("i"), fa(
  " در فضای تصویر و بُعدِ هم‌شماره در فضای متن درباره‌ی یک چیز سخن " +
  "می‌گویند. آن فرض همان چیزی است که بازیابیِ صفر-شات می‌سنجد، و همان " +
  "چیزی که برای زبانی که رمزگذار ندیده برقرار نیست. معادله‌ی ("), en("2"),
  fa(
  ") چنین فرضی ندارد: الحاق هیچ تناظری میان دو فضا نمی‌خواهد و تصمیم را " +
  "به سرِ رده‌بند وامی‌گذارد. همین تفاوت است که پیش‌بینی می‌کند در " +
  "ناهم‌ترازیِ کامل، الحاق باید "), fa("بهتر", { bold: true }), fa(
  " باشد — و جدول ۳ همین را نشان می‌دهد."),
]));

// a roadmap paragraph for §1
const roadmap = [];
roadmap.push(P([fa(
  "ساختار ادامه چنین است. بخش ۲ کارهای پیشین را مرور می‌کند. بخش ۳ " +
  "گونه‌های ترکیب را صوری تعریف می‌کند و سنجه‌ی هم‌ترازی را شرح می‌دهد. " +
  "بخش ۴ داده و پیکربندی را می‌آورد. بخش ۵ نتایج را در نُه زیربخش گزارش " +
  "می‌کند. بخش ۶ پیش‌بینی‌های فارسی را از رابطه‌ی به‌دست‌آمده استخراج " +
  "می‌کند، بخش ۷ محدودیت‌ها را می‌شمارد و بخش ۸ جمع‌بندی می‌کند. بخش‌های " +
  "۹ و ۱۰ به تکرارپذیری و ملاحظات اخلاقی می‌پردازند."),
]));

module.exports = { fusionMath, roadmap };
