#!/usr/bin/python

import json
import html
import os
import sys
from pathlib import Path

# ==========================================
# Configuration
# ==========================================

# Change this to the path of your JSONL text file.
#DATA_FILE = Path("/var/www/arbsq/projects/speech/data.jsonl")

#DATA_FILE = Path("/var/www/arbsq/user_data/speech_project/response.txt")
cur_path="/var/www/arbsq/user_data/speech_project/responses.txt"
DATA_FILE = Path(cur_path)

# Keys to extract. Use dots for nested JSON keys.
COLUMNS = [
    ("participantId", "participantId"),
    ("optionId", "optionId"),
    ("question.audio_path", "audio_path"),
    ("question.original_sentence", "original_sentence"),
    ("consented", "consented"),
]

# ==========================================
# Helpers
# ==========================================

def get_nested_value(obj, key):
    """Retrieve a possibly nested value using dot notation."""
    value = obj

    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return ""

        value = value[part]

    if value is None:
        return ""

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)

    if isinstance(value, bool):
        return "true" if value else "false"

    return str(value)


def esc(value):
    """Escape values for safe HTML output."""
    return html.escape(str(value), quote=True)


def read_jsonl(filepath):
    """Read one JSON object per line."""
    records = []
    errors = []

    with filepath.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            # Ignore blank lines
            if not line:
                continue

            try:
                record = json.loads(line)

                if not isinstance(record, dict):
                    raise ValueError("Expected a JSON object")

                records.append(record)

            except (json.JSONDecodeError, ValueError) as exc:
                errors.append((line_number, str(exc)))

    return records, errors


# ==========================================
# Generate HTML
# ==========================================

def main():
    print("Content-Type: text/html; charset=utf-8")
    print()

    try:
        records, errors = read_jsonl(DATA_FILE)

    except OSError:
        # Avoid exposing filesystem paths or internal details.
        print("""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Data Viewer</title></head>
<body><h1>Unable to read the data file</h1>
<p>Check the configured file path and its read permissions.</p>
</body></html>""")
        return

    print("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JSON Data Viewer</title>

<style>
    * { box-sizing: border-box; }

    body {
        margin: 0;
        padding: 32px 20px;
        background: #f3f6fa;
        color: #243247;
        font-family: Arial, Helvetica, sans-serif;
    }

    .container {
        max-width: 1200px;
        margin: 0 auto;
        background: #fff;
        border-radius: 12px;
        padding: 28px;
        box-shadow: 0 4px 18px rgba(30, 50, 80, 0.08);
    }

    h1 {
        margin: 0 0 8px;
        color: #17365d;
        font-size: 26px;
    }

    .subtitle {
        margin: 0 0 24px;
        color: #64748b;
    }

    .stats {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-bottom: 22px;
    }

    .stat {
        background: #eef4ff;
        border: 1px solid #dce8fb;
        border-radius: 8px;
        padding: 12px 18px;
    }

    .stat strong {
        display: block;
        color: #17365d;
        font-size: 22px;
    }

    .stat span {
        color: #64748b;
        font-size: 12px;
    }

    .table-wrap {
        overflow-x: auto;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }

    thead {
        background: #17365d;
        color: white;
    }

    th, td {
        text-align: left;
        padding: 14px 16px;
        border-bottom: 1px solid #e8edf3;
        vertical-align: top;
        overflow-wrap: anywhere;
    }

    th {
        font-weight: 600;
        white-space: nowrap;
    }

    tbody tr:nth-child(even) {
        background: #f8fafc;
    }

    tbody tr:hover {
        background: #edf4ff;
    }

    .index {
        color: #64748b;
        width: 60px;
    }

    .empty {
        text-align: center;
        padding: 36px;
        color: #64748b;
    }

    .notice {
        margin-top: 20px;
        padding: 14px 16px;
        background: #fff7e6;
        border: 1px solid #f5d89a;
        border-radius: 8px;
        color: #754c00;
    }

    .notice ul {
        margin-bottom: 0;
    }

    @media (max-width: 600px) {
        body { padding: 12px; }
        .container { padding: 16px; }
        h1 { font-size: 22px; }
        th, td { padding: 10px; }
    }
</style>
</head>

<body>
<div class="container">
    <h1>JSON Data Viewer</h1>
    <p class="subtitle">
        Selected fields extracted from the JSON Lines data file.
    </p>
""")

    print('<div class="stats">')
    print(
        '<div class="stat"><strong>{}</strong>'
        '<span>Records loaded</span></div>'.format(len(records))
    )
    print(
        '<div class="stat"><strong>{}</strong>'
        '<span>Invalid lines</span></div>'.format(len(errors))
    )
    print('</div>')

    print('<div class="table-wrap">')
    print('<table>')
    print('<thead><tr><th class="index">#</th>')

    for _, title in COLUMNS:
        print("<th>{}</th>".format(esc(title)))

    print('</tr></thead><tbody>')

    if not records:
        colspan = len(COLUMNS) + 1
        print(
            '<tr><td colspan="{}" class="empty">'
            'No valid records found.</td></tr>'.format(colspan)
        )

    else:
        for row_number, record in enumerate(records, start=1):
            print("<tr>")
            print('<td class="index">{}</td>'.format(row_number))

            for key, _ in COLUMNS:
                value = get_nested_value(record, key)
                print("<td>{}</td>".format(esc(value)))

            print("</tr>")

    print('</tbody></table></div>')

    if errors:
        print('<div class="notice">')
        print("<strong>Some lines could not be parsed</strong>")
        print("<ul>")

        # Limit displayed errors to avoid an excessively long page.
        for line_number, message in errors[:20]:
            print(
                "<li>Line {}: {}</li>".format(
                    line_number, esc(message)
                )
            )

        if len(errors) > 20:
            print(
                "<li>And {} additional errors.</li>".format(
                    len(errors) - 20
                )
            )

        print("</ul></div>")

    print("""
</div>
</body>
</html>
""")


if __name__ == "__main__":
    main()
