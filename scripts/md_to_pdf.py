"""Convert markdown files to PDFs via markdown + weasyprint.

Optional tooling (not in requirements.txt): pip install markdown weasyprint

Usage: md_to_pdf.py <input.md> [<input2.md> ...]
Writes <input>.pdf alongside each input file.
"""
import sys
from pathlib import Path

import markdown
from weasyprint import HTML, CSS

CSS_STYLE = """
@page {
    size: A4;
    margin: 2.0cm 1.8cm;
    @bottom-right {
        content: counter(page) " / " counter(pages);
        font-family: 'DejaVu Sans', sans-serif;
        font-size: 9pt;
        color: #666;
    }
}
body {
    font-family: 'DejaVu Sans', sans-serif;
    font-size: 10.5pt;
    line-height: 1.45;
    color: #222;
}
h1 {
    font-size: 20pt;
    color: #222;
    margin-top: 0;
    margin-bottom: 0.4em;
    border-bottom: 2px solid #444;
    padding-bottom: 0.15em;
}
h2 {
    font-size: 14pt;
    color: #222;
    margin-top: 1.2em;
    margin-bottom: 0.4em;
    border-bottom: 1px solid #bbb;
    padding-bottom: 0.1em;
}
h3 {
    font-size: 11.5pt;
    color: #333;
    margin-top: 0.9em;
    margin-bottom: 0.3em;
}
p, li { orphans: 3; widows: 3; }
code {
    font-family: 'DejaVu Sans Mono', monospace;
    font-size: 9.5pt;
    background: #f2f2f2;
    padding: 1px 4px;
    border-radius: 2px;
}
pre {
    background: #f2f2f2;
    padding: 0.5em;
    border-radius: 3px;
    font-size: 9.5pt;
    overflow-x: auto;
}
pre code { background: none; padding: 0; }
table {
    border-collapse: collapse;
    margin: 0.5em 0 1em 0;
    width: 100%;
    font-size: 10pt;
}
th, td {
    border: 1px solid #ccc;
    padding: 5px 8px;
    text-align: left;
    vertical-align: top;
}
th {
    background: #eee;
    font-weight: 600;
}
tr:nth-child(even) td { background: #fafafa; }
ul, ol { padding-left: 1.4em; }
strong { color: #000; }
hr { border: 0; border-top: 1px solid #ccc; margin: 1em 0; }
"""


def convert(in_path: Path):
    text = in_path.read_text()
    html_body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    html = f"<!doctype html><html><head><meta charset='utf-8'></head><body>{html_body}</body></html>"
    out_path = in_path.with_suffix(".pdf")
    HTML(string=html).write_pdf(out_path, stylesheets=[CSS(string=CSS_STYLE)])
    print(f"{in_path}  →  {out_path}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        convert(Path(arg))
