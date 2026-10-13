
# Dialogue behaviour generator

Deterministic synthetic conversation generator for teaching a small GPT-2-style
assistant elementary dialogue policy, memory, appropriate topic follow-ups and
natural termination. Pure Python standard library. No internet or downloaded models.

## Quick start

```bash
python generator.py --out data --train 2000 --validation 200 --test 200 --seed 42
python compile_chatml.py --data data --out data/chatml
python -m unittest discover -v
```

Default corpus: 2,000 training episodes, 200 validation, 200 test.

## Files

- `generator.py`: simulated user, dialogue state, action policy, response realizer, episode generation, validation and export.
- `compile_chatml.py`: converts three views to `{prompt, completion}` ChatML JSONL. Compute language-model loss **only on `completion`** (including end marker). For GPT-2, properly register and train special tokens if using them.
- `test_generator.py`: regression tests.
- `data/{split}.episodes.jsonl`: canonical episodes, hidden user state, actions, observations and state transitions.
- `data/{policy,response,history,extraction}/{split}.jsonl`: supervised views.
- `data/chatml/{policy,response,history}/{split}.jsonl`: ready-to-tokenize prompt/completion pairs.
- `data/manifest.json`: generation parameters and row counts.

## View semantics

`policy`: history -> dialogue action. `response`: history + oracle action -> response.
`history`: history -> response (the important autonomous end-to-end baseline).
`extraction`: prior history + user utterance -> structured observation delta.

No hidden simulator facts are fed to `history` or `policy` by default.
The dataset is English; metadata is in JSON.

## Design limitations

- This is a controlled **baseline**, not a general-purpose conversation engine.
- The current response realization uses fixed templates and covers three slots (time of day, activity, mood), plus several follow-up topics. It does **not** guarantee naturalness or robustness of GPT-2 trained on it.
- Test splits use the same template families as train, so test is **not** an out-of-distribution evaluation. Add paraphrase and adversarial sets before claiming generalization.
- No semantic parsing of arbitrary real user language. Simulated observations are oracle labels from the user simulator.
- Conversation lengths are bounded; that is not proof of no repetition in model-driven free-running chat.
- This code does not itself train a GPT-2; use an external SFT trainer and correctly mask prompts.

## Rebuilding from source

Delete `data/` and run the commands above. All generated files are reproducible
with the same Python version and parameters.


### How to run

```bash
cd dialogue_behaviour_v01_rebuilt

python -m unittest discover -v

python generator.py --out data --train 2000 --validation 200 --test 200 --seed 42

python compile_chatml.py --data data --out data/chatml
```

