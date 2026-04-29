#!/bin/bash
cd "$(dirname "$0")"
tectonic main.tex
cp main.pdf ai4good_v1.pdf
echo "PDF built: main.pdf"
echo "Versioned PDF: ai4good_v1.pdf"
