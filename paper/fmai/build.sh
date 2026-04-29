#!/bin/bash
cd "$(dirname "$0")"
tectonic main.tex
cp main.pdf fmai_v1.pdf
echo "PDF built: main.pdf"
echo "Versioned PDF: fmai_v1.pdf"
