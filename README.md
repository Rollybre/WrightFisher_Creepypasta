Code, data, and article source for the paper on fandom data and Wright-Fisher models.

## Structure

```
.
├── code/
│   ├── wf.ipynb            # Wright-Fisher simulations
│   ├── analysis_stats.py   # Power-law MLE fit + KS goodness-of-fit tests
│   └── test.ipynb          # Data exploration
├── data/
│   ├── fandom_data.csv     # Main fandom dataset
│   └── fandom_links.csv    # Fandom links
└── article/
    ├── main.tex            # LaTeX source
    ├── camera_ready.tex    # Camera-ready version
    ├── references.bib      # Bibliography
    ├── anthology-ch.cls    # ACL Anthology class file
    ├── main.pdf            # Compiled paper
    └── illustration/       # All figures
```

## Running the code

```bash
# Statistical analysis (requires numpy, matplotlib, pandas, powerlaw, scipy)
python code/analysis_stats.py

# Jupyter notebooks
jupyter notebook code/wf.ipynb
```
