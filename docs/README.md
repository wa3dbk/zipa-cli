# Documentation

`manual.tex` is the comprehensive **ZIPA-CLI User Manual** (LaTeX). Build it to PDF:

```bash
cd docs
make            # -> manual.pdf  (uses latexmk if present, else pdflatex twice)
make clean      # remove build artifacts
```

Requires a TeX distribution (TeX Live / MacTeX / MiKTeX). The manual covers
installation, the model registry, every input source and output format, batching,
timestamped alignment, the comparison mode, the web viewer, and the ONNX length
caveat.

For day-to-day usage start with the top-level [`README.md`](../README.md); for
runnable, end-to-end walkthroughs see the notebooks in [`tutorials/`](../tutorials/).
