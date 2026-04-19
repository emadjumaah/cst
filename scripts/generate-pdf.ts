/**
 * Generates professional academic PDF papers from Markdown sources.
 * Uses marked for Markdown→HTML conversion, MathJax for equation rendering,
 * and Puppeteer (system Chrome) for HTML→PDF.
 *
 * Usage:
 *   npx tsx scripts/generate-pdf.ts
 */

import puppeteer from "puppeteer-core";
import { marked } from "marked";
import { readFileSync, writeFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
import { tmpdir } from "os";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = resolve(__dirname, "..");

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

// ---------------------------------------------------------------------------
// Markdown pre-processing
// ---------------------------------------------------------------------------

/**
 * Protect LaTeX math blocks from marked's inline parser.
 * Replaces $...$ and $$...$$ with placeholders, restores after marked runs.
 */
function protectMath(md: string): {
  protected: string;
  map: Map<string, string>;
} {
  const map = new Map<string, string>();
  let i = 0;
  const result = md
    // Display math first ($$...$$)
    .replace(/\$\$([\s\S]*?)\$\$/g, (_, inner) => {
      const key = `MATHBLOCK${i++}MATHBLOCK`;
      map.set(key, `\\[${inner}\\]`);
      return key;
    })
    // Inline math ($...$)
    .replace(/\$([^\n$]+?)\$/g, (_, inner) => {
      const key = `MATHINLINE${i++}MATHINLINE`;
      map.set(key, `\\(${inner}\\)`);
      return key;
    });
  return { protected: result, map };
}

function restoreMath(html: string, map: Map<string, string>): string {
  let result = html;
  for (const [key, value] of map) {
    result = result.replaceAll(key, value);
  }
  return result;
}

// ---------------------------------------------------------------------------
// HTML template
// ---------------------------------------------------------------------------

function buildHTML(markdownContent: string, rtl: boolean): string {
  const { protected: safeMd, map } = protectMath(markdownContent);
  const rawHTML = marked.parse(safeMd) as string;
  const bodyHTML = restoreMath(rawHTML, map);

  const lang = rtl ? "ar" : "en";
  const dir = rtl ? "rtl" : "ltr";
  const textAlign = rtl ? "right" : "justify";
  const borderSide = rtl ? "right" : "left";
  const listPadding = rtl ? "padding-right: 22pt;" : "padding-left: 22pt;";

  const fontImport = rtl
    ? `@import url('https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;0,700;1,400;1,700&display=swap');`
    : `@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400&family=JetBrains+Mono:wght@400;500&display=swap');`;

  const bodyFont = rtl
    ? `'Amiri', 'Noto Naskh Arabic', 'Geeza Pro', Georgia, serif`
    : `'Source Serif 4', Georgia, 'Times New Roman', serif`;

  const monoFont = `'JetBrains Mono', 'Courier New', monospace`;

  return `<!DOCTYPE html>
<html lang="${lang}" dir="${dir}">
<head>
  <meta charset="UTF-8">
  <script>
    window.MathJax = {
      tex: {
        inlineMath:   [['\\\\(', '\\\\)']],
        displayMath:  [['\\\\[', '\\\\]']],
        processEscapes: true
      },
      options: {
        skipHtmlTags: ['script','noscript','style','textarea','pre','code']
      },
      startup: { typeset: true }
    };
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
  <style>
    ${fontImport}

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: ${bodyFont};
      font-size: 11pt;
      line-height: 1.65;
      color: #111;
      direction: ${dir};
      text-align: ${textAlign};
      background: #fff;
    }

    .paper {
      max-width: 162mm;
      margin: 0 auto;
    }

    /* ── Title & author ─────────────────────────────────────────── */
    h1 {
      font-size: 17pt;
      font-weight: 700;
      text-align: center;
      line-height: 1.35;
      margin-bottom: 6pt;
    }

    /* The bold author byline paragraph that follows h1 */
    h1 ~ p:first-of-type {
      text-align: center;
      font-size: 11pt;
      font-style: italic;
      color: #333;
      margin-bottom: 18pt;
    }

    hr {
      border: none;
      border-top: 1px solid #bbb;
      margin: 16pt 0;
    }

    /* ── Section headings ───────────────────────────────────────── */
    h2 {
      font-size: 13pt;
      font-weight: 700;
      margin-top: 22pt;
      margin-bottom: 7pt;
      padding-bottom: 3pt;
      border-bottom: 1.5px solid #999;
      page-break-after: avoid;
    }

    h3 {
      font-size: 11.5pt;
      font-weight: 700;
      margin-top: 14pt;
      margin-bottom: 5pt;
      page-break-after: avoid;
    }

    h4 {
      font-size: 11pt;
      font-weight: 700;
      margin-top: 10pt;
      margin-bottom: 4pt;
    }

    /* ── Body text ──────────────────────────────────────────────── */
    p {
      margin-bottom: 7pt;
      orphans: 3;
      widows: 3;
    }

    strong { font-weight: 700; }
    em     { font-style: italic; }

    ul, ol {
      margin: 6pt 0 8pt 0;
      ${listPadding}
    }

    li { margin-bottom: 3pt; }

    /* ── Blockquote ─────────────────────────────────────────────── */
    blockquote {
      border-${borderSide}: 3pt solid #888;
      margin: 10pt 0;
      padding-${borderSide}: 14pt;
      color: #444;
      font-style: italic;
    }

    /* ── Code ───────────────────────────────────────────────────── */
    code {
      font-family: ${monoFont};
      font-size: 8.8pt;
      background: #f3f3f3;
      padding: 1pt 4pt;
      border-radius: 2pt;
      white-space: nowrap;
    }

    pre {
      background: #f3f3f3;
      border-${borderSide}: 3pt solid #777;
      padding: 9pt 12pt;
      margin: 9pt 0 10pt;
      overflow-x: auto;
      direction: ltr;
      text-align: left;
      page-break-inside: avoid;
    }

    pre code {
      background: none;
      padding: 0;
      font-size: 8.8pt;
      line-height: 1.5;
      white-space: pre;
    }

    /* ── Tables ─────────────────────────────────────────────────── */
    table {
      border-collapse: collapse;
      width: 100%;
      margin: 10pt 0 12pt;
      font-size: 9.2pt;
      page-break-inside: avoid;
    }

    th {
      background: #eee;
      font-weight: 700;
      border: 1px solid #bbb;
      padding: 4pt 8pt;
      text-align: ${textAlign};
    }

    td {
      border: 1px solid #bbb;
      padding: 3.5pt 8pt;
      vertical-align: top;
    }

    tr:nth-child(even) td { background: #f9f9f9; }

    /* ── Math ───────────────────────────────────────────────────── */
    mjx-container { font-size: 10.5pt !important; }

    /* ── Page setup ─────────────────────────────────────────────── */
    @page {
      size: A4;
      margin: 25mm 25mm 22mm 25mm;
    }
  </style>
</head>
<body>
  <div class="paper">
    ${bodyHTML}
  </div>
</body>
</html>`;
}

// ---------------------------------------------------------------------------
// PDF generation
// ---------------------------------------------------------------------------

async function generatePDF(
  inputPath: string,
  outputPath: string,
  rtl: boolean,
): Promise<void> {
  console.log(`  Reading  : ${inputPath}`);
  const markdown = readFileSync(inputPath, "utf-8");
  const html = buildHTML(markdown, rtl);

  // Write HTML to a temp file so Chrome can load external resources
  const tmpFile = resolve(tmpdir(), `cst-paper-${Date.now()}.html`);
  writeFileSync(tmpFile, html, "utf-8");

  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });

  try {
    const page = await browser.newPage();
    await page.goto(`file://${tmpFile}`, {
      waitUntil: "networkidle0",
      timeout: 60_000,
    });

    // Let MathJax finish typesetting
    await page
      .waitForFunction("window.MathJax?.startup?.promise", { timeout: 20_000 })
      .catch(() => {});
    await page
      .evaluate(async () => {
        if ((window as any).MathJax?.startup?.promise) {
          await (window as any).MathJax.startup.promise;
        }
      })
      .catch(() => {});

    // Small settle time for fonts
    await new Promise((r) => setTimeout(r, 1500));

    await page.pdf({
      path: outputPath,
      format: "A4",
      printBackground: true,
      displayHeaderFooter: false,
      margin: { top: "25mm", right: "25mm", bottom: "22mm", left: "25mm" },
    });

    console.log(`  ✓ Output : ${outputPath}`);
  } finally {
    await browser.close();
  }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

async function main() {
  const papers: Array<{
    input: string;
    output: string;
    rtl: boolean;
    label: string;
  }> = [
    {
      label: "English",
      input: resolve(root, "docs/cst-paper.md"),
      output: resolve(root, "docs/cst-paper.pdf"),
      rtl: false,
    },
    {
      label: "Arabic",
      input: resolve(root, "docs/cst-paper-ar.md"),
      output: resolve(root, "docs/cst-paper-ar.pdf"),
      rtl: true,
    },
  ];

  for (const paper of papers) {
    console.log(`\nGenerating ${paper.label} paper…`);
    await generatePDF(paper.input, paper.output, paper.rtl);
  }

  console.log("\nDone. Both PDFs written to docs/");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
