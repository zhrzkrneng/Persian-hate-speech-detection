// Generate the manuscript. Every number comes from runs/paper_numbers.json,
// which is computed from the committed per-seed records; nothing is typed.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, ShadingType,
  SectionType, ImageRun, PageNumber, Footer,
} = require("docx");
const { SIZE, half, fa, en } = require("./style.js");

const N = JSON.parse(fs.readFileSync(process.argv[2] || "../runs/paper_numbers.json", "utf8"));
const OUT = process.argv[3] || "maghale.docx";

// ---------- small helpers ----------
const fmt = (x, d = 2) => (x === null || x === undefined ? "—" : x.toFixed(d));
const signed = (x, d = 2) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);
// Persian digits, because the body text is Persian and the journal's own
// template sets its numbers that way. Statistics inside tables stay Latin so
// a reader can compare them with the English literature.
const faDigits = (s) => String(s).replace(/[0-9]/g, (d) => "۰۱۲۳۴۵۶۷۸۹"[d]);

function P(runs, opts = {}) {
  const {
    align = AlignmentType.JUSTIFIED, spacing = { after: 100, line: 276 },
    indent = { firstLine: 283 }, keepNext = false,
  } = opts;
  return new Paragraph({
    bidirectional: true, alignment: align, spacing, indent, keepNext,
    children: runs.map((r) => new TextRun(r)),
  });
}

function H(text, level) {
  const size = level === 1 ? SIZE.h1 : level === 2 ? SIZE.h2 : SIZE.h3;
  return new Paragraph({
    bidirectional: true,
    alignment: AlignmentType.START,
    spacing: { before: 220, after: 110 },
    keepNext: true,
    children: [new TextRun(fa(text, { size, bold: true }))],
  });
}

function caption(text) {
  return new Paragraph({
    bidirectional: true, alignment: AlignmentType.CENTER,
    spacing: { before: 90, after: 90 },
    children: [new TextRun(fa(text, { size: SIZE.caption, bold: true }))],
  });
}

// `ltr` is for a cell holding a number. In a right-to-left paragraph a leading
// sign is reordered to the far side -- "-1.53" renders as "1.53-" and "n.s."
// as ".n.s" -- so numeric cells are laid out left-to-right while the text
// cells around them stay right-to-left.
function cell(runs, { header = false, width, align = AlignmentType.CENTER,
                      ltr = false } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: header ? { type: ShadingType.CLEAR, fill: "E8E8E8" } : undefined,
    margins: { top: 40, bottom: 40, left: 60, right: 60 },
    children: [new Paragraph({
      bidirectional: !ltr, alignment: align, spacing: { after: 0 },
      children: runs.map((r) => new TextRun(r)),
    })],
  });
}

// The same problem inside a sentence: a signed number or a p-value in running
// Persian text needs an explicit left-to-right mark to keep its sign.
const LRM = "\u200E";
const ltr = (s) => LRM + s + LRM;

// A table sized to one column of the two-column body.
const COL = 4600;
function table(widths, rows) {
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    visuallyRightToLeft: true,
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

const T = (t, o = {}) => fa(t, { size: SIZE.tableBody, ...o });
const TL = (t, o = {}) => en(t, { size: SIZE.tableBody, ...o });

module.exports = { P, H, caption, cell, table, T, TL, COL, fmt, signed, faDigits, ltr, N, OUT,
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, AlignmentType,
  WidthType, SectionType, ImageRun, Footer, PageNumber, fa, en, SIZE, half, fs, path };
