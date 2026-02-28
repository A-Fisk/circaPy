# Changelog

All notable changes to circaPy are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.1.6] - 2026-02-28

### Added
- Full documentation website via Sphinx and Read the Docs
  - API reference auto-generated from NumPy docstrings for all modules
  - Quickstart guide with code examples
  - `pydata-sphinx-theme` matching the scientific Python ecosystem
  - LaTeX math rendering via MathJax (used in `calculate_IS` and `calculate_TV`)
  - Intersphinx links to NumPy, pandas, and matplotlib docs
- `.readthedocs.yaml` configuration for Read the Docs deployment
- `make docs` target to build documentation locally

### Changed
- `calculate_IV` and `calculate_IS` now return per-column values instead of a single scalar
- All figure-plotting functions now gated behind `showfig` keyword argument

### Fixed
- Resolved `UnboundLocalError` in `plot_activity_profile`
- `plot_activity_profile` now correctly loops over multiple PIR columns

### Performance
- Actogram rendering significantly faster: early numpy array extraction, disabled interactive rendering during loop
- Matplotlib font cache used in CI to speed up test runs

### Infrastructure
- Switched from conda to `uv` for environment and dependency management
- Switched from autopep8 to `ruff` for formatting
- Upgraded to Python 3.12
- GitHub Actions CI now uses `uv`; tests only run on pull requests
- Input validation decorator (`validate_input`) now rejects NaN values and invalid DatetimeIndex frequencies
- Added Ayobami Fawole as co-author

---

## [0.1.5] - 2024-11-29

### Changed
- Renamed project from actiPy to circaPy
