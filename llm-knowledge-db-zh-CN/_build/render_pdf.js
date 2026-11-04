// Render a local HTML file to a print-ready A4 PDF with Chromium (Playwright).
// Usage: node render_pdf.js <in.html> <out.pdf> [header-title] [--plain]
//   --plain : no running header/footer and zero page margins (used for the cover).
const path = require('path');
const { chromium } = require('playwright');

(async () => {
  const args = process.argv.slice(2);
  const plain = args.includes('--plain');
  const [inHtml, outPdf, headerTitle = ''] = args.filter((a) => a !== '--plain');
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('file://' + path.resolve(inHtml), { waitUntil: 'load' });
  await page.evaluate(() => document.fonts.ready);
  const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
  const font = "font-family:'Noto Sans CJK SC',sans-serif;font-size:7.5pt;color:#888;";
  const opts = {
    path: outPdf,
    format: 'A4',
    printBackground: true,
    outline: !plain,
    tagged: true,
  };
  if (plain) {
    opts.margin = { top: '0', bottom: '0', left: '0', right: '0' };
  } else {
    opts.displayHeaderFooter = true;
    opts.headerTemplate = `<div style="${font}width:100%;padding:0 20mm;text-align:right;">${esc(headerTitle)}</div>`;
    opts.footerTemplate = `<div style="${font}width:100%;text-align:center;">— <span class="pageNumber"></span> —</div>`;
    opts.margin = { top: '20mm', bottom: '18mm', left: '20mm', right: '20mm' };
  }
  await page.pdf(opts);
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
