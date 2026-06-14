#!/usr/bin/env python3
"""
Build a review PDF of the paper (paper.md).

Pipeline: paper.md -> HTML (Python-Markdown, tables + extras) -> PDF (WeasyPrint).
The result figures referenced in paper.md are base64-embedded so the PDF is
self-contained, and the page background is painted opaque white (margins
included) via `@page { background }` so the PDF reads correctly in dark-mode
viewers as well as light ones.

Dependencies:
    pip install markdown weasyprint
    # WeasyPrint needs the native Pango/Cairo libraries:
    #   macOS (Homebrew):  brew install pango
    #   Debian/Ubuntu:     apt-get install libpango-1.0-0 libpangocairo-1.0-0
# On macOS the Homebrew libraries live under /opt/homebrew/lib; if WeasyPrint
# cannot find them this script re-execs itself once with
# DYLD_FALLBACK_LIBRARY_PATH set, so a plain `python3 make_pdf.py` just works.

Usage:
    python3 make_pdf.py            # writes paper.pdf next to this script
"""
import os
import sys
import re
import base64

HERE = os.path.dirname(os.path.abspath(__file__))


def _import_weasyprint():
    """Import WeasyPrint, re-exec'ing with the Homebrew lib path on macOS if the
    native Pango/Cairo libraries are not on the default loader search path."""
    try:
        from weasyprint import HTML
        return HTML
    except OSError:
        if sys.platform == "darwin" and "DYLD_FALLBACK_LIBRARY_PATH" not in os.environ:
            os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = "/opt/homebrew/lib:/usr/local/lib"
            os.execv(sys.executable, [sys.executable] + sys.argv)
        raise


CSS = """
@page { size: Letter; margin: 0.9in 0.95in; background: #ffffff; }
html, body { background: #ffffff; }
body { font-family: Georgia, "Times New Roman", serif; font-size: 10.5pt;
       line-height: 1.5; color: #111; text-align: justify; hyphens: auto; }
h1 { font-size: 19pt; text-align: center; line-height: 1.25; margin: 0 0 0.15em;
     hyphens: none; }
h1 + p { text-align: center; color: #444; margin: 0 0 1.2em; }
h2 { font-size: 13.5pt; margin: 1.25em 0 0.4em; border-bottom: 1px solid #ccc;
     padding-bottom: 2px; page-break-after: avoid; hyphens: none; }
h3 { font-size: 11.5pt; margin: 1em 0 0.3em; page-break-after: avoid; hyphens: none; }
p { margin: 0 0 0.6em; }
strong { font-weight: 700; }
code { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 0.88em;
       background: #f2f2f2; padding: 0 2px; border-radius: 3px; }
table { border-collapse: collapse; width: 100%; font-size: 8.6pt;
        margin: 0.6em 0 1.1em; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3px 6px; text-align: left;
         vertical-align: top; hyphens: none; }
th { background: #eee; font-weight: 700; }
img { display: block; margin: 0.7em auto 0.25em; max-width: 78%;
      page-break-inside: avoid; }
ol, ul { margin: 0 0 0.7em; padding-left: 1.5em; }
li { margin-bottom: 0.35em; }
"""


def build(md_path=None, pdf_path=None):
    import markdown

    md_path = md_path or os.path.join(HERE, "paper.md")
    pdf_path = pdf_path or os.path.join(HERE, "paper.pdf")

    HTML = _import_weasyprint()  # fail fast (and re-exec on macOS) before doing work

    with open(md_path) as f:
        md = f.read()
    html_body = markdown.markdown(md, extensions=["extra", "sane_lists"])

    # Embed any figure PNGs referenced from the run directory as base64 data
    # URIs, so the PDF carries its own images.
    def embed(m):
        with open(os.path.join(HERE, m.group(1)), "rb") as fh:
            data = base64.b64encode(fh.read()).decode()
        return 'src="data:image/png;base64,%s"' % data

    html_body, n = re.subn(r'src="(outputs/[^"]+\.png)"', embed, html_body)

    html = ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            "<style>%s</style></head><body>%s</body></html>" % (CSS, html_body))

    HTML(string=html, base_url=HERE).write_pdf(pdf_path)
    print("embedded %d figure(s) -> %s" % (n, pdf_path))


if __name__ == "__main__":
    build()
