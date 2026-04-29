#!/bin/bash
cd "$(dirname "$0")"
tectonic main.tex
cp main.pdf aiwild_v1.pdf
echo "PDF built: main.pdf"
echo "Versioned PDF: aiwild_v1.pdf"
