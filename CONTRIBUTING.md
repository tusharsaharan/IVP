# Working together

## Before editing

1. Read your section in `docs/TEAM.md` and the contracts in `docs/CONTRACTS.md`.
2. Pull the latest main branch.
3. Create a short feature branch, for example `sahil/verified-kau-loader` or `aryan/connectivity-qc`.
4. Keep changes within your owned module unless you coordinate the handoff with Tushar and Vivek.

```bash
git switch main
git pull --ff-only
git switch -c your-name/short-task
```

## Handoff requirements

Every contribution includes an importable implementation, one working command, input/output details, and any dependency changes. For scientific changes include the relevant config, provenance, and method description. Run `python -m pytest` and `python -m ruff check .` before opening a pull request.

Do not independently change the class mapping, feature order, fold generation, channel map, or bundle schema. Coordinate these changes because they affect other modules. Do not place raw EEG, model bundles, local paths, credentials, or generated synthetic metrics in commits.

## Finish a contribution

```bash
git add src/asd_eeg/YOUR_MODULE tests/YOUR_TEST.py
git commit -m "Describe the specific change"
git push -u origin your-name/short-task
```

Open a pull request against `main`. Mention the owner of the next pipeline stage. Tushar integrates the application and Vivek reviews model/interface changes.

## Documents

Each member provides their own technical subsection, figure captions, and commands to Pranav. All reported metrics must be traceable to a completed run under `results/`. Preserve run IDs and configuration IDs in the report.
