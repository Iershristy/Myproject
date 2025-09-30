# IEEE Manuscript (LaTeX)

This folder contains the IEEE double-column LaTeX manuscript for:

- Title: Attention-Driven Deep Learning for Parkinson's Disease Severity Assessment from Gait Time Series
- Venue style: IEEEtran (conference)

## Files
- `main.tex`: Main LaTeX source
- `results.tex`: Results-focused PDF with tables and graphs
- `ieee_pd_gait.tex`: IEEE-formatted paper with your provided content (paraphrased)

## Quick Build (local)
Requires a LaTeX distribution with `latexmk` and `pdflatex`.

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error docs/paper/main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error docs/paper/results.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error docs/paper/ieee_pd_gait.tex
```

The output PDF will be `docs/paper/main.pdf`.

If you do not have LaTeX installed, on Ubuntu you can run:

```bash
debian_frontend=noninteractive sudo apt-get update && sudo apt-get install -y texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra latexmk
```

## GitHub Actions (auto-build)
A workflow is provided at `.github/workflows/latex.yml` that builds the PDFs on every push. After the run finishes, download the artifacts named `paper-pdf`, `results-pdf`, and `ieee-pd-gait-pdf` from the workflow page.

## Notes
- Replace the placeholder author block in `main.tex` with your details.
- Add figures and tables under this folder and `\includegraphics` them as needed.
- If you prefer BibTeX, you can switch to a `.bib` file and `\bibliographystyle{IEEEtran}` + `\bibliography{refs}`.