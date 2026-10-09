# Budgeted Search for AI4Science

Experiment-search code accompanying **[Prior or Feedback? What an LLM Uses When Adapting Neural Operators](https://arxiv.org/abs/2610.12325)**,
accepted to the **NeurIPS 2026 AI4Science Workshop**.

The scaffold provides four controllers: an LLM, Random, TPE, and an LLM warm-start followed by TPE. You supply
an experiment and its search space. The core imports only the standard library; `optuna` and `openai` are optional
extras loaded inside their controllers.

The LLM controller sees the full history of configurations and validation feedback, proposes the next
configuration, and selects the medoid from an ensemble of proposals. The repository includes an offline example,
configurable prompts, run recording and replay, and a standalone script for verifying the paper's reported results
from separately supplied run records.

## Install

```bash
uv sync --frozen                                  # core only
uv sync --frozen --extra optuna --extra llm       # Random, TPE, hybrid and the OpenAI-compatible transport
uv sync --frozen --extra optuna --extra llm --group dev   # adds pytest and ruff
```

## Run the offline example

```bash
uv run --frozen python examples/minimal/run.py --output runs/my-example
uv run --frozen python scripts/summarise.py runs/my-example
```

The example minimises a deterministic objective over a float, a log-scale float and a categorical parameter. By
default it replays five fixed proposals. No API key or dataset is needed. It demonstrates the interface only.

```bash
uv run --frozen --extra optuna python examples/minimal/run.py --controller tpe        # or random
export LLM_API_KEY=...                           # name configurable with --api-key-env
uv run --frozen --extra optuna --extra llm python examples/minimal/run.py \
  --controller llm --model <model> --base-url <endpoint> --ensemble-k 3
uv run --frozen --extra optuna --extra llm python examples/minimal/run.py \
  --controller hybrid --model <model> --warmstart-trials 2
```

Existing run directories are never overwritten.

## Integrate your experiment

```python
from pathlib import Path

from budgeted_search.contracts import Observation
from budgeted_search.controllers.replay import ReplayController
from budgeted_search.runner import RunConfig, Runner
from budgeted_search.space import Parameter, SearchSpace


class MyExperiment:
    def run_trial(self, parameters, context):
        # Train with context.seed (run seed plus trial index) so the recorded seed is the one used.
        score, artefact = train_and_validate(parameters, context)
        return Observation(score, artefact, feedback={"validation_loss": score})

    def evaluate_selected(self, artefact):
        # Evaluate the selected artefact on held-out data, without retraining.
        return evaluate_test(artefact)


space = SearchSpace((Parameter("lr", "float", "Learning rate", low=1e-5, high=1e-2, log=True),))
config = RunConfig(10, 42, "Describe the task", score_name="validation loss")
Runner(config, space).run(
    MyExperiment(),
    ReplayController([{"lr": 1e-3}]),
    Path("runs/my-experiment"),
    project_root=Path.cwd(),
    metadata={"experiment": "my-experiment"},
)
```

`train_and_validate` and `evaluate_test` are yours. The experiment owns its model, data, splits, loss and evaluation.

`Observation` fields:

- `score`: the selection score. Any real number except `bool`; non-finite scores are never selected.
- `artefact`: a reference (usually a path) passed to `evaluate_selected` for the selected trial.
- `metrics`: recorded only. The controller never sees them.
- `feedback`: shown to the controller in the next prompt. Include only quantities that are legitimate to use for
  selection, never held-out results.
- `costs`: realised costs, recorded only. The runner adds `wall_seconds`.

Values may be JSON types, array scalars and arrays from numerical libraries (anything with a `tolist` method), and
non-finite floats, which are stored as `{"nonfinite": "nan"}`. An observation value that cannot be serialised, an
out-of-range or incomplete proposal, or a trial exception becomes a failed trial. `RunConfig.on_trial_error` decides whether the run then aborts (default) or continues.
Failed trials count against the budget. Selection is the best finite score, first wins ties, and `direction` is
`minimize` or `maximize`. The held-out evaluation runs once, on the selected artefact.

## LLM controller

`LLMController(space, transport, ...)` takes any callable mapping chat messages to a `Response`. `OpenAITransport`
(extra `llm`) wraps an OpenAI-compatible endpoint, keeps the key on the client only, defaults to SDK
`max_retries=0` because the controller owns retries, and captures `reasoning_content` when the provider returns it.

- **Prompts.** Three templates ship as package data: `system.md`, `initial.md` and `iteration.md`, with named
  `{placeholders}`. The iteration prompt carries all prior trials condensed, the previous configuration and the
  controller's own reasoning for it, the observed feedback, the best score so far, a direction-aware improvement
  (positive means a new best for both `minimize` and `maximize`) and the strategy distribution. To customise:

  ```python
  from budgeted_search.controllers.llm import LLMController
  from budgeted_search.controllers.prompting import PromptTemplates

  PromptTemplates.default().write_to(Path("my_prompts"))  # edit the copies
  LLMController(space, transport, templates=PromptTemplates.from_dir(Path("my_prompts")))
  ```

  A template that references an unknown placeholder raises an error. Escape literal braces as `{{` and `}}`.
- **Response format.** One JSON object with `reasoning`, `strategy` (`explore` or `exploit`) and one key per
  parameter. Fenced or prose-wrapped JSON is accepted. Integer parameters accept integral floats. Out-of-range
  values are rejected and re-sampled, never clamped. The names `reasoning` and `strategy` are reserved.
- **Thinking traces.** Provider reasoning is logged in the audit and never placed in a prompt. Only the `reasoning`
  field of the JSON reply is carried forward.
- **Ensemble medoid.** With `ensemble_k > 1` the members are sampled in parallel threads (`parallel=True`, so the
  transport must be thread-safe) and the proposal with the smallest summed distance to the others is chosen, ties
  to the lowest member index. The default `SearchSpace.distance` is `mixed_distance`: numeric parameters use the
  absolute difference on a unit scale (log10 for `log=True`), categorical parameters 0 or 1, averaged over
  parameters. Zero-width numeric parameters are excluded so gated constants do not dilute distances. Pass
  `SearchSpace(..., distance=fn)` for your own geometry.
- **Retries.** Each member gets `max_attempts` (default 3) covering transport errors, empty content and invalid
  proposals, with waits of `min(max, min * 2**(attempt - 1))` seconds from `retry_wait=(1.0, 60.0)`. A member that
  exhausts its attempts is dropped and recorded. The run proceeds with the survivors unless `require_complete=True`
  (raises `IncompleteEnsembleError`) or none survive (raises `RuntimeError`).
- **Provider identity lock.** The first non-missing served model and fingerprint are locked. A later different
  value raises `ServedModelChangedError` immediately and is never retried, since pooling responses from different
  models would mix controllers. A missing field does not establish stability.
- **Audit.** `controller.last_audit` and `Proposal.audit`, also written to `events.jsonl`, hold the rendered
  messages, every attempt per member (outcome, error, content, model, fingerprint, usage, reasoning), the parsed
  candidates, selection statistics (survivors, distance sums, mean and max pairwise distance) and the chosen
  reasoning and strategy. Credentials are never included.

## Hybrid controller

`HybridController(warmstart, tpe, warmstart_trials=N)` runs the warm-start controller for the first `N` trials, adds
each observation to an `OptunaController` configured for TPE, then hands over. Set the TPE startup trial count
explicitly (the example uses 0). During warm-start the LLM prompt includes a note stating how many LLM-controlled
proposals remain before TPE takes over. The budget must contain at least one TPE trial.

## Replay

```bash
uv run --frozen python examples/minimal/run.py \
  --replay-log runs/my-example/events.jsonl --output runs/replayed-example
```

Replay repeats recorded parameters. It does not reproduce what a live LLM would answer.

## Generated files

Each run creates a new directory, by default under the gitignored `runs/`:

```text
config.json              # Run configuration and described search space
manifest.json            # Code version, package versions, hardware and your metadata
events.jsonl             # Proposals with audits, trial results and the final selection
trial_0000/result.json   # Observation, costs and error for each trial, beside your artefacts
summary.json             # Selected trial and held-out results on success
failure.json             # Failure cause and completed trials on failure
analysis/trials.csv      # Written by scripts/summarise.py
```

Keep credentials out of task descriptions, feedback and metadata. Resuming an interrupted run is not implemented.

## Paper results

The run records behind the paper's reported results are published as a separate dataset (DOI: [10.5281/zenodo.23213042](https://doi.org/10.5281/zenodo.23213042)).
`paper/verify_claims.py` recomputes the paper's numbers from them; see [`paper/README.md`](paper/README.md).

## Not included

- No models, datasets, training code, checkpoints or figures. The paper's FNO/PDEBench training integration is not
  part of this release.
- Replay does not reproduce live LLM behaviour.

## Tests and lint

```bash
uv run --frozen --group dev pytest
uv run --frozen --group dev ruff check .
uv run --frozen --group dev ruff format --check .
```

Add `--extra optuna --extra llm` to `uv run` to include the tests that need those extras; they skip otherwise.

## Citation

If you use this code in research, please cite **Prior or Feedback? What an LLM Uses When Adapting Neural Operators**
(accepted to the NeurIPS 2026 AI4Science Workshop; [arXiv:2610.12325](https://arxiv.org/abs/2610.12325)).

```bibtex
@misc{chan2026prior,
  title         = {Prior or Feedback? What an LLM Uses When Adapting Neural Operators},
  author        = {Chan, Julian and Mora Jimenez, Javier},
  year          = {2026},
  eprint        = {2610.12325},
  archivePrefix = {arXiv},
  note          = {NeurIPS 2026 Workshop on AI for Science}
}
```

[`CITATION.cff`](CITATION.cff) contains the software citation metadata.
