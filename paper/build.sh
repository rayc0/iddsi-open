#!/usr/bin/env bash
# W12 - build IDDSI-Open technical note with pandoc.
# Produces paper.html and, if a PDF engine is available, paper.pdf.
# v0: classifier/flow result cells report the locked 2026-09-07 numbers;
# this script changes no content.
set -uo pipefail
cd "$(dirname "$0")"

if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc not found - install pandoc (brew install pandoc) and re-run." >&2
  exit 1
fi

# HTML (always built; requires no LaTeX)
# NOTE (W12): references.bib is loaded via the `bibliography:` field in
# paper.md's YAML front matter. Do NOT pass references.bib as a positional
# input - pandoc then parses the raw "@misc{...}" lines as markdown citations
# and leaks the bib source text into the document body.
pandoc paper.md \
  --from markdown+yaml_metadata_block \
  --citeproc \
  --toc --toc-depth=2 \
  --standalone \
  -o paper.html
html_rc=$?
echo "paper.html exit=$html_rc"

# PDF (only if a TeX engine exists)
pdf_engine=""
for e in pdflatex xelatex lualatex tectonic typst; do
  if command -v "$e" >/dev/null 2>&1; then pdf_engine="$e"; break; fi
done
if [ -n "$pdf_engine" ]; then
  # typst is not a LaTeX engine; pandoc --pdf-engine handles it directly.
  # -M keywords= (empty) suppresses the keywords line: pandoc 3.9's default
  # typst template emits YAML keywords unquoted as typst code, which fails
  # to compile. Keywords remain in paper.md's YAML and in the HTML output.
  pandoc paper.md \
    --from markdown+yaml_metadata_block \
    --citeproc \
    --toc --toc-depth=2 \
    --pdf-engine "$pdf_engine" \
    -M keywords= \
    -o paper.pdf
  echo "paper.pdf exit=$? (engine: $pdf_engine)"
else
  echo "No PDF engine (pdflatex/xelatex/lualatex/tectonic/typst) found - PDF build skipped; HTML build above stands."
fi

exit "$html_rc"
