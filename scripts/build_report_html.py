from pathlib import Path
import sys

import markdown


CSS = """
@page {
  size: A4;
  margin: 18mm 16mm;
}

* {
  box-sizing: border-box;
}

body {
  color: #172033;
  font-family: "Segoe UI", Arial, sans-serif;
  font-size: 11.5pt;
  line-height: 1.6;
  margin: 0 auto;
  max-width: 920px;
}

h1 {
  border-bottom: 3px solid #1f6feb;
  color: #0f172a;
  font-size: 27pt;
  line-height: 1.15;
  margin: 0 0 18px;
  padding-bottom: 12px;
}

h2 {
  color: #1f3b69;
  font-size: 17pt;
  margin: 28px 0 10px;
  page-break-after: avoid;
}

h3 {
  color: #334155;
  font-size: 13.5pt;
  margin: 20px 0 8px;
  page-break-after: avoid;
}

p {
  margin: 8px 0;
}

hr {
  border: 0;
  border-top: 1px solid #d7dee8;
  margin: 22px 0;
}

table {
  border-collapse: collapse;
  font-size: 10.5pt;
  margin: 12px 0 18px;
  page-break-inside: avoid;
  width: 100%;
}

th {
  background: #edf4ff;
  color: #0f172a;
  font-weight: 700;
}

th,
td {
  border: 1px solid #cbd5e1;
  padding: 8px 10px;
  text-align: left;
  vertical-align: top;
}

code {
  background: #f1f5f9;
  border-radius: 4px;
  color: #0f172a;
  font-family: Consolas, "Courier New", monospace;
  font-size: 9.5pt;
  padding: 1px 4px;
}

pre {
  background: #0f172a;
  border-radius: 8px;
  color: #e2e8f0;
  font-family: Consolas, "Courier New", monospace;
  font-size: 9.5pt;
  line-height: 1.45;
  overflow-wrap: break-word;
  padding: 12px 14px;
  white-space: pre-wrap;
}

pre code {
  background: transparent;
  color: inherit;
  padding: 0;
}

ul {
  margin: 8px 0 12px 22px;
  padding: 0;
}

li {
  margin: 4px 0;
}

a {
  color: #1f6feb;
  text-decoration: none;
}
"""


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python scripts/build_report_html.py REPORT_EN.md REPORT_EN.html")

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    md_text = input_path.read_text(encoding="utf-8")

    body = markdown.markdown(
        md_text,
        extensions=["extra", "tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Quran Ayah Correction Chatbot Report</title>
  <style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
