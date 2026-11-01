// The plain-language report, as its own Word file. Single column, same RTL
// machinery and the same numbers file as the manuscript, so the report and
// the paper cannot disagree.
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, WidthType, BorderStyle, ShadingType, Footer, PageNumber,
} = require("docx");
const { SIZE, half, fa, en } = require("./style.js");

const N = JSON.parse(fs.readFileSync("../runs/paper_numbers.json", "utf8"));
const M = N.multilingual;
const fmt = (x, d = 2) => (x === null || x === undefined ? "—" : Number(x).toFixed(d));
const sg = (x, d = 2) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);
const LRM = "‎";
const ltr = (s) => LRM + s + LRM;
const ben = (e) => N.fusion_benefit[e];
const ret = (e, l) => N.retrieval[e]?.[l]?.t2i_r1;
const tok = (m, l) => (N.tokenisation.rows || []).find(r => r.model === m && r.language === l);

function P(runs, opts = {}) {
  return new Paragraph({
    bidirectional: true,
    alignment: opts.align || AlignmentType.JUSTIFIED,
    spacing: { after: 120, line: 300 },
    children: runs.map(r => new TextRun(r)),
  });
}
function H(text, level = 1) {
  return new Paragraph({
    bidirectional: true, alignment: AlignmentType.START,
    spacing: { before: level === 1 ? 320 : 220, after: 120 }, keepNext: true,
    children: [new TextRun(fa(text, { size: level === 1 ? 16 : 13, bold: true }))],
  });
}
function bullet(text) {
  return new Paragraph({
    bidirectional: true, alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 90 }, indent: { start: 340, hanging: 170 },
    children: [new TextRun(fa("• " + text))],
  });
}
function cell(runs, { header = false, width, align = AlignmentType.CENTER, ltrCell = false } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: header ? { type: ShadingType.CLEAR, fill: "EAEAEA" } : undefined,
    margins: { top: 60, bottom: 60, left: 80, right: 80 },
    children: [new Paragraph({
      bidirectional: !ltrCell, alignment: align, spacing: { after: 0 },
      children: runs.map(r => new TextRun(r)),
    })],
  });
}
function table(widths, rows) {
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths, visuallyRightToLeft: true,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 6 },
      bottom: { style: BorderStyle.SINGLE, size: 6 },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2 },
      insideVertical: { style: BorderStyle.NONE },
    },
    rows,
  });
}
const T = (t, o = {}) => fa(t, { size: 10, ...o });
const TL = (t, o = {}) => en(t, { size: 10, ...o });

module.exports = { Document, Packer, Paragraph, TextRun, TableRow, AlignmentType,
  Footer, PageNumber, P, H, bullet, cell, table, T, TL, fa, en, SIZE, fmt, sg, ltr,
  ben, ret, tok, N, M, fs };
