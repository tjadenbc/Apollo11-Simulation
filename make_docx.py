#!/usr/bin/env python3
"""
Build a Microsoft Word (.docx) version of the paper (paper.md).

Pipeline: paper.md -> HTML (Python-Markdown, tables + extras) -> DOCX (pandoc).
The figures referenced from the run directory are embedded by pandoc (it reads
them via --resource-path), so the .docx is self-contained, and the markdown
pipe tables and the one raw-HTML table (Table 4) both become native Word tables.

Dependencies:
    pip install markdown
    # plus pandoc on PATH:
    #   macOS (Homebrew):  brew install pandoc
    #   Debian/Ubuntu:     apt-get install pandoc

Usage:
    python3 make_docx.py            # writes TJADEN_2026_06_15.docx next to this script
"""
import os
import sys
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
STEM = "TJADEN_2026_06_15"  # output filename stem (author_date) for the generated .docx

# Light styling so Word gets bordered tables and a centered title/author block.
CSS = """
body { font-family: 'Times New Roman', serif; font-size: 11pt; line-height: 1.4; }
h1 { font-size: 18pt; text-align: center; }
h1 + p { text-align: center; }
h2 { font-size: 13pt; } h3 { font-size: 11.5pt; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; }
th, td { border: 1px solid #888888; padding: 3px 6px; vertical-align: top; text-align: left; }
th { background: #eeeeee; }
img { max-width: 100%; }
"""


def _find_pandoc():
    for c in (shutil.which("pandoc"), "/opt/homebrew/bin/pandoc", "/usr/local/bin/pandoc"):
        if c and os.path.exists(c):
            return c
    sys.exit("pandoc not found — install it (e.g. `brew install pandoc`).")


def build(md_path=None, docx_path=None):
    import markdown

    md_path = md_path or os.path.join(HERE, "paper.md")
    docx_path = docx_path or os.path.join(HERE, STEM + ".docx")
    pandoc = _find_pandoc()

    with open(md_path) as f:
        md = f.read()
    body = markdown.markdown(md, extensions=["extra", "sane_lists"])
    html = ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            "<style>%s</style></head><body>%s</body></html>" % (CSS, body))

    # pandoc reads HTML from stdin; --resource-path lets it find/embed the PNGs.
    subprocess.run(
        [pandoc, "-f", "html", "-t", "docx", "--resource-path", HERE, "-o", docx_path],
        input=html.encode("utf-8"), check=True,
    )
    print("wrote %s" % docx_path)


if __name__ == "__main__":
    build()
