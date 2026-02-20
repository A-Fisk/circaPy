[![Main build](https://github.com/A-Fisk/circaPy/actions/workflows/python-ci-uv.yml/badge.svg?branch=main)](https://github.com/A-Fisk/circaPy/actions/workflows/python-ci-uv.yml)

[![Development build](https://github.com/A-Fisk/circaPy/actions/workflows/python-ci-uv.yml/badge.svg?branch=development)](https://github.com/A-Fisk/circaPy/actions/workflows/python-ci-uv.yml)
# CircaPy

CircaPy is a python module for circadian analysis of activity data.
It was developed using laboratory
rodents data but is applicable across species and monitoring devices.

A limited version is available as an [interactive
website](https://circapywebsite.streamlit.app/)


## Getting Started

Install circapy from pip

```
pip install circapy
```

This will install circaPy in your current python environment.
Package dependencies are listed in the pyproject.toml file


## Using circaPy

circaPy provides a set of functions to analyse and plot the most common
methods of circadian analysis.

Create some test data 
```
import pandas as pd
import numpy as np
import circaPy.activity as act

# Create a sample dataset with time-series activity data
index = pd.date_range(start='2024-01-01', periods=86400, freq="10s")
values = np.random.randint(0, 100, size=(len(index),2))

df = pd.DataFrame(values, index=index)
``` 

Calculate IV 
```
# Use circaPy's calculate_IV function to compute Interdaily Variability
iv = act.calculate_IV(df)

# Print the result
print(f"Interdaily Variability (IV). Col 0: {iv[0]:.4f}")
```

Plot actogram 
```
# Use circaPy plot_actogram
import circaPy.plots as cpp

cpp.plot_actogram(df, showfig=True)
```

## Contributing 

1. Fork this repository
2. Create branch `git checkout -b <branch-name>`
3. Create uv environment `uv sync`
4. Use uv to run test suite with `make test`
4. Make your changes and commit them `git commit -m <commit-message>`
5. PR back to the `development` branch
    - ensure tests are passing, will be required to merge into
      development branch.

## Authors  

- Angus Fisk 
    - [angus_fisk@hotmail.com](angus_fisk@hotmail.com)

## Licence 

- Available under GNU general public licence 




