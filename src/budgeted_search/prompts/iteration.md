Trial {previous_trial} results:
{note}

**Previous configuration**:
```json
{previous_config}
```

**Previous reasoning**: {previous_reasoning}

**Observed feedback**:
{observed_feedback}

**Context**:
- Best `{score_name}` so far: {best_summary}
- Improvement over the earlier best: {improvement} (positive = new best, negative = worse, zero = tie or no earlier trial)
- Trials remaining, including the one you are about to propose: {trials_remaining}

**History** (all {history_length} prior trials, condensed; strategy distribution below):
{history_summary}

Respond with a JSON object only.
