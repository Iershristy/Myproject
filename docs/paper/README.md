# IEEE Manuscript (LaTeX)

This folder contains the IEEE double-column LaTeX manuscript for:

- Title: Attention-Driven Deep Learning for Parkinson's Disease Severity Assessment from Gait Time Series
- Venue style: IEEEtran (conference)

## Files
- `ieee_pd_gait.tex`: IEEE-formatted paper (all content consolidated)
- `results.tex`: Results-focused PDF with tables and graphs

## Quick Build (local)
Requires a LaTeX distribution with `latexmk` and `pdflatex`.

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error docs/paper/ieee_pd_gait.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error docs/paper/results.tex
```

The output PDF will be `docs/paper/main.pdf`.

If you do not have LaTeX installed, on Ubuntu you can run:

```bash
debian_frontend=noninteractive sudo apt-get update && sudo apt-get install -y texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra latexmk
```

## GitHub Actions (auto-build)
A workflow is provided at `.github/workflows/latex.yml` that builds the PDFs on every push. After the run finishes, download the artifacts named `ieee-pd-gait-pdf` and `results-pdf` from the workflow page.

## Notes
- Replace the placeholder author block in `main.tex` with your details.
- Add figures and tables under this folder and `\includegraphics` them as needed.
- If you prefer BibTeX, you can switch to a `.bib` file and `\bibliographystyle{IEEEtran}` + `\bibliography{refs}`.