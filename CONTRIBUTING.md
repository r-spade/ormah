# Contributing to Ormah

Help improve an integration, clarify a guide, or report a reproducible bug. Use [GitHub Issues](https://github.com/r-spade/ormah/issues) for problems and proposals, or the [Ormah Discord](https://discord.gg/guBU6XweBu) for discussion.

## Development

Use Python 3.11+, Node.js/npm, and Make. Run the following in an activated Python virtual environment with pip available:


```bash
git clone https://github.com/r-spade/ormah.git
cd ormah
make install
python -m pytest
```

### Retrieval evals

Two golden-corpus eval harnesses measure retrieval quality from a source checkout:

```bash
make eval                       # whisper + recall evals with fail-below quality bars
ormah eval whisper run          # whisper pipeline eval (--show-failures, --category, --fail-below)
ormah eval recall run           # recall/retrieval eval (--fail-below)
ormah eval whisper mine         # mine provisional cases from your live whisper_log (read-only)
ormah eval whisper import-labels  # confirm mined labels after human review
```

Eval corpora are intentionally **local-only and gitignored**: they seed real
memories from the developer's own usage, so they never ship in the repo. See
`eval/whisper/corpus/README.md` for the case-design and honest-labeling rules.
CI runs tests and lint only; eval bars are enforced locally via `make eval`.

## Release Process


Releases are published through the manual GitHub Actions `Release` workflow. The workflow
is guarded so only the GitHub login listed in `RELEASE_ALLOWED_ACTOR` can run the release
path.

Before running the workflow:

1. Bump `pyproject.toml`.
2. Bump `integrations/claude-plugin/.claude-plugin/plugin.json` to the same version.
3. Merge the release PR to `main`.

Then open GitHub Actions, run `Release` from `main`, and enter the version without the
leading `v`, for example `0.12.0`.

The workflow verifies that the requested version matches `pyproject.toml`, verifies that
the Claude plugin manifest has the same version, runs the Python test suite with
`ORMAH_LLM_PROVIDER=none`, builds the UI, builds one wheel, publishes that wheel to PyPI,
creates `v<version>`, and creates the GitHub Release with generated notes.

PyPI publishing uses Trusted Publishing. No long-lived PyPI token is stored in the repo,
local `.env`, or developer machines. The PyPI project should trust this repository,
`.github/workflows/release.yml`, and the `pypi` GitHub environment.

Protect the `pypi` GitHub environment with the release manager as the required reviewer.
Do not enable prevent-self-review unless a second reviewer is intentionally required for
every release. If Trusted Publishing is not configured yet, use a project-scoped PyPI API
token only as a temporary environment-scoped GitHub Actions secret.


Use the GitHub Actions release workflow; do not publish releases through local scripts or the local Makefile release target.
