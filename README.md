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



##  Parameters 


|  Parameters | Description   | Value  |   |   |
|---|---|---|---|---|
|  $N$| Nombre de classe  |  40 |   |   |
|  $n_i$ | Population initiale  |  1000 |   |   |
|  $n_f$ |   Population finale| 2000  |   |   |
|  $T$ | Nombre de génération  | 1000  |   |   |
|  $t$| Fenêtre d'analyse  |  [0;300] |   |   |
|  $\alpha$ | Taux d'archivage  | 0.05  |   |   |
|  (new)$b$ |Conformité (vs proba génération précédente) |[0;1]   |   |   |
|  (new)$c$ | Le contenu (à voir)  |   |   |   |



## Descriptive Stats

|   |   |   |   |
|---|---|---|---|
| KS  |   |   |   |
| MLE |   |   |   |
|Nombre de Hill (Shanon en plus ?)(diversité intra)   ||   |   |
|Gini (richesse intra) |||||
|tx de remplacement (topx)||||
