const B = require("./build.js");
const { P, H, caption, cell, table, T, TL, N, TableRow, AlignmentType, fa, en, SIZE } = B;

// A reader comparing the two baselines has to hold six differences in mind;
// the table holds them instead.
const compare = [];
compare.push(caption("جدول (۱۰): تفاوت دو معماری مرجع، همان‌طور که اینجا پیاده شده‌اند"));
const W = [1300, 1650, 1650];
const ROWS = [
  ["رمزگذار", "منجمد", "منجمد"],
  ["ترکیب", "الحاق یا عنصری", "عنصری"],
  ["وفق‌دهنده", "ندارد", "پس‌ماندی روی هر مُد"],
  ["نرمال‌سازی", "ندارد", "L2 پیش از ترکیب"],
  ["رده‌بند", "خطی", "کسینوسی با ضریب مقیاس"],
];
compare.push(table(W, [
  new TableRow({ tableHeader: true, children: [
    cell([T("مؤلفه", { bold: true })], { header: true, width: W[0] }),
    cell([TL("Hate-CLIPper", { bold: true })], { header: true, width: W[1] }),
    cell([TL("MemeCLIP", { bold: true })], { header: true, width: W[2] }),
  ]}),
  ...ROWS.map((r) => new TableRow({ children: [
    cell([T(r[0])], { width: W[0], align: AlignmentType.START }),
    cell([T(r[1])], { width: W[1] }),
    cell([T(r[2])], { width: W[2] }),
  ]})),
]));
compare.push(P([fa(
  "هر دو رمزگذار را منجمد نگه می‌دارند، پس تفاوتشان تنها در سرِ رده‌بند " +
  "است و مقایسه‌شان در این مطالعه مقایسه‌ی همان سه افزوده است و نه چیز " +
  "دیگری."),
], { spacing: { before: 140 } }));

// future work
const future = [];
future.push(H("۸-۱ کارهای آینده", 2));
future.push(P([fa(
  "سه گام، به‌ترتیبِ آنچه بیشترین چیز را روشن می‌کند. نخست، افزودنِ " +
  "نقاطِ بیشتر به محور: پیکره‌ای از میم‌های موازی در چند زبان اجازه " +
  "می‌دهد همین رابطه به‌جای چهار نقطه روی بیست نقطه برازش شود، و تنها " +
  "آن‌گاه است که می‌توان درباره‌ی "), fa("شکلِ", { bold: true }), fa(
  " رابطه ادعایی کرد. دوم، سنجه‌ای از هم‌ترازیِ بُعد‌به‌بُعد که از بازیابی " +
  "به سازوکارِ ضربِ عنصری نزدیک‌تر باشد. سوم، و تنها راهِ آزمودنِ " +
  "پیش‌بینی‌های بخش ۶، ساختِ یک مجموعه‌ی ارزیابیِ فارسیِ برچسب‌خورده " +
  "با پروتکلی که مرزهای حاشیه‌نویسی را صریح کرده باشد."),
], { indent: { firstLine: 0 } }));
future.push(P([fa(
  "این سومی کاری است که این پژوهش نتوانسته انجام دهد و پنهانش نمی‌کند: " +
  "حاشیه‌نویسِ آموزش‌دیده لازم دارد، و نبودنش دلیلِ آن است که فارسی در " +
  "این مقاله پیش‌بینی است و نه نتیجه."),
]));

module.exports = { compare, future };
