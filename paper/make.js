const B = require("./build.js");
const { Document, Packer, Paragraph, TextRun, AlignmentType, SectionType,
        Footer, PageNumber, fs, N, OUT, fa, en, SIZE } = B;
const A = require("./content_a.js");
const Bc = require("./content_b.js");
const C = require("./content_c.js");
const D = require("./content_d.js");
const E = require("./content_e.js");

const footer = new Footer({
  children: [new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ children: [PageNumber.CURRENT], size: 18,
      font: { ascii: "Times New Roman", hAnsi: "Times New Roman" } })],
  })],
});

// A4 with the usual journal margins; the front matter spans the full width and
// the body runs in two columns, joined by a continuous section break so they
// share one page.
const page = {
  margin: { top: 1134, right: 1134, bottom: 1134, left: 1134 },
};

const notice = [];
if (N.provisional && N.provisional.length) {
  notice.push(new Paragraph({
    bidirectional: true, alignment: AlignmentType.CENTER,
    spacing: { after: 120 },
    children: [new TextRun(fa(
      `پیش‌نویس: ${N.provisional.length} سلول از این جدول‌ها از اجرای پیشین ` +
      `برداشته شده و بازاجرای آن‌ها در جریان است.`,
      { size: SIZE.caption, bold: true }))],
  }));
}

const doc = new Document({
  styles: { default: { document: { run: {
    font: { ascii: "Times New Roman", hAnsi: "Times New Roman", cs: "B Lotus" },
    size: 22,
  }}}},
  sections: [
    { properties: { page, column: { count: 1 } }, footers: { default: footer },
      children: [...notice, ...A.front] },
    { properties: { type: SectionType.CONTINUOUS, page,
        column: { count: 2, space: 400, equalWidth: true } },
      footers: { default: footer },
      children: [...A.body, ...Bc.body, ...C.body, ...E.afterResults,
                 ...D.body] },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log(`wrote ${OUT} (${(buf.length / 1024).toFixed(0)} KB)`);
  if (N.provisional && N.provisional.length) {
    console.log(`PROVISIONAL cells: ${N.provisional.length}`);
  }
});
