# IEEE Manuscript (LaTeX)

This folder contains the IEEE double-column LaTeX manuscript for:

- Title: Attention-Driven Deep Learning for Parkinson's Disease Severity Assessment from Gait Time Series
- Venue style: IEEEtran (conference)

## Files
- `main.tex`: Main LaTeX source

## Quick Build (local)
Requires a LaTeX distribution with `latexmk` and `pdflatex`.

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error docs/paper/main.tex
```

The output PDF will be `docs/paper/main.pdf`.

If you do not have LaTeX installed, on Ubuntu you can run:

```bash
debian_frontend=noninteractive sudo apt-get update && sudo apt-get install -y texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra latexmk
```

## GitHub Actions (auto-build)
A workflow is provided at `.github/workflows/latex.yml` that builds the PDF on every push. After the run finishes, download the artifact named `paper-pdf` from the workflow page to get the compiled PDF.

## Notes
- Replace the placeholder author block in `main.tex` with your details.
- Add figures and tables under this folder and `\includegraphics` them as needed.
- If you prefer BibTeX, you can switch to a `.bib` file and `\bibliographystyle{IEEEtran}` + `\bibliography{refs}`.