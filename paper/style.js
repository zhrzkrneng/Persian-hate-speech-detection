// Fonts and sizes from the journal's own specification table (جدول ۱ of the
// template). Persian text is B Lotus; Latin text is Times New Roman one point
// smaller at every position. Sizes here are half-points, as docx requires.
const FA = "B Lotus";
const EN = "Times New Roman";

// position -> Persian point size. The Latin size is this minus one.
const SIZE = {
  title: 15,
  authors: 13,
  h1: 14,
  h2: 13,
  h3: 12,
  abstract: 12,
  footnote: 10,
  caption: 10,
  tableBody: 9,
  refFa: 11,
  refEn: 10,
  body: 11,
};

const half = (pt) => Math.round(pt * 2);

// A Persian run. `size` is the Persian point size; the Latin font inside the
// same run is set one point smaller, which is what the specification asks for
// and what a run-level font pair expresses.
function fa(text, opts = {}) {
  const { size = SIZE.body, bold = false, italics = false } = opts;
  return {
    text,
    bold,
    italics,
    rightToLeft: true,
    size: half(size),
    font: { ascii: EN, hAnsi: EN, cs: FA },
    // Complex-script size is separate in OOXML and is the one that applies to
    // Persian; without it Word uses its own default and the table above is
    // silently ignored.
    szCs: half(size),
  };
}

// A Latin run inside Persian text: one point smaller, left-to-right.
function en(text, opts = {}) {
  const { size = SIZE.body, bold = false, italics = false } = opts;
  return {
    text,
    bold,
    italics,
    rightToLeft: false,
    size: half(size - 1),
    font: { ascii: EN, hAnsi: EN, cs: EN },
    szCs: half(size - 1),
  };
}

module.exports = { FA, EN, SIZE, half, fa, en };
