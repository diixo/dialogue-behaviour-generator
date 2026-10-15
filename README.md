# Dialogue behaviour generator

Deterministic synthetic English dialogues for learning memory, dialogue actions,
topic follow-ups and appropriate termination. Default corpus: 2,000/200/200 episodes
for train/validation/test. The trainer is a scaffold; no trained model is included.

## Data

| View | Input → target |
| --- | --- |
| `extraction` | History + state before the user message + user message → observation and complete updated state |
| `policy` | History + updated state → next action |
| `response` | History + updated state + action → assistant reply |
| `history` | History → assistant reply |

Within an episode, `extraction[t].target_state` equals the `state` of
`policy[t+1]` and `response[t+1]`; `history[t+1]` is the corresponding direct-reply
example. `next_turn` records this link. The initial assistant turn has no extraction.
State contains observed facts and dialogue memory, never undisclosed simulator facts.

`data/` stores JSONL; `json/` contains equivalent JSON arrays and metadata.
`*.episodes.*` are canonical episodes; `sft/` holds structured `messages/completion`
exports. Chat formatting is applied at runtime, with plain text as the default.

## Run

```bash
python -m pip install -r requirements.txt
python generator.py --out data --train 2000 --validation 200 --test 200 --seed 42
python compile_sft.py --data data --out data/sft --json-out json/sft
python -m unittest discover -v
```

The generator writes raw JSONL; refresh raw JSON copies when regenerating the corpus.
Generation and SFT export themselves require only the Python standard library.

`data.py` provides individual datasets, `DialogueChainDataset` and `chain_collate_batch`
for four aligned task batches. `compile_sft.build_messages` also accepts predicted
state/action at inference. Training uses reference states/actions; inference must
use model predictions. Loss covers completion and EOS; prompts and padding are masked.

Templates are shared across splits. Evaluate new phrasings and model-driven
conversations separately before drawing conclusions about generalization.
