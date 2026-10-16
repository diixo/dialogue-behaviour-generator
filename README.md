# Dialogue behaviour generator

Deterministic synthetic English dialogues for learning memory, dialogue actions,
topic follow-ups and appropriate termination. Default corpus: 2,000/200/200 episodes
for train/validation/test. All training modes default to pretrained `gpt2`; no trained checkpoint is included.


#### Run django-server:
```bash
py ui/manage.py migrate
py ui/manage.py runserver
```

Start the job worker in a second terminal: `py ui/manage.py runworker`.

The UI browses episodes, linked SFT examples, training artifacts and benchmark errors.
It reads `data/`, `outputs/` and `runs/`; paths are configured in `ui/ui/settings.py`.
The UI queues training/evaluation, shows live progress and supports checkpoint stop/resume.
Chat with a trained UBAR model through the same worker and inspect each turn's state and action.

Commands: `/reset`, `/exit`; use `--device cuda` for GPU. Only `torch` and `transformers`
are required. To distribute the chat separately, copy `chat_runtime.py`, `inference.py`,
`dialogue_protocol.py`, `dialogue_format.py`, `requirements-chat.txt` and the model directory
(including tokenizer files and `experiment.json`).


## Data

| View | Input → target |
| --- | --- |
| `extraction` | History + state before the user message + user message → observation and complete updated state |
| `policy` | History + updated state → next action |
| `response` | History + updated state + action → assistant reply |
| `history` | History → assistant reply |

Within an episode, `extraction[t].target_state` equals the `state` of
`policy[t]` and `response[t]`; `history[t]` is the corresponding direct-reply
example. `next_turn` records the linked assistant turn. The user always starts;
extraction processes the first message from empty memory before any assistant action.
State contains observed facts and dialogue memory, never undisclosed simulator facts.

Each `.jsonl` is accompanied by a `.json` array with identical records and `indent=2`.
`*.episodes.*` are canonical episodes; `sft/` holds structured `messages/completion`
exports. Chat formatting is applied at runtime, with plain text as the default.

## Run
```bash
python trainer.py --task ubar --epochs 2 --output outputs
```

Full lifecycle:
```bash
python -m pip install -r requirements.txt
python generator.py --out data --train 2000 --validation 200 --test 200 --seed 42
python compile_sft.py --data data --out data/sft
python trainer.py --task ubar --output runs/ubar
python benchmark.py --model runs/ubar --chain full --context reference --user-source dataset --output runs/eval-ubar
python -m unittest discover -v
```

Generation and SFT export themselves require only the Python standard library.

Training tasks: `dst` (state only), `extraction`, `policy`, `response`, `history`, `ubar`.
Benchmark settings: `--chain` selects stages (`full` = DST → policy → response),
`--context reference|predicted` selects prior history/memory, and
`--user-source dataset|simulator` selects user messages. Predicted context requires
the full chain; simulated users require predicted context. Defaults: full/reference/dataset.
Reports group errors by reaction, action, target and depth; see `--help` for options.

`data.py` provides individual datasets, `DialogueChainDataset` and `chain_collate_batch`
for four aligned task batches. `compile_sft.build_messages` also accepts predicted
state/action at inference. Training uses reference states/actions; inference must
use model predictions. Loss covers completion and EOS; prompts and padding are masked.

Templates are shared across splits. Evaluate new phrasings and model-driven
conversations separately before drawing conclusions about generalization.

### INFO

У каждого примера должна быть понятная модели инструкция:

```text
Извлеки наблюдение из сообщения.
Выбери следующее действие.
Сформулируй ответ для заданного действия.
Продолжи диалог.
```

Если нам нужна явная зависимость, как в SimpleTOD, обучающий пример должен выглядеть иначе:

```text
Вход: история и последнее сообщение пользователя

Цель:
наблюдение: ...
действие: ...
ответ: ...
```

В наших данных есть важная особенность для связанных групп:
```text
extraction[episode_id, t]
          ↕
policy[episode_id, t]
response[episode_id, t]
history[episode_id, t]
```

`extraction[t]` разбирает сообщение пользователя перед действием и ответом ассистента на том же ходе `t`, включая самый первый запрос.


Я бы сравнил три эксперимента:
1. Только history — контрольный результат.
2. history + policy — проверка пользы явного обучения выбору действия.
3. Все четыре задачи — проверка дополнительной пользы извлечения и ответа по действию.

Сравнивать их следует при сопоставимом бюджете обучения, а оценивать прежде всего:
- повторные вопросы об уже известном;
- повторные запросы после отказа;
- реакцию на просьбу завершить разговор;
- уместность продолжения темы;
- поведение в диалоге с собственными предыдущими ответами модели.


Последний пункт особенно хорошо мотивирован экспериментами UBAR. Для `response` также нужна проверка с `предсказанным` действием: правильное действие из датасета позволяет отдельно оценить формулирование ответа, но скрывает ошибки выбора.

EOS здесь отвечает за завершение отдельной генерации. Решение, `когда завершать разговор`, относится к поведению и должно проверяться отдельно.

Для следующего обсуждения я бы взял `PPTOD` как основу смешанного обучения, а `SimpleTOD/UBAR` — как альтернативу с явными промежуточными шагами. Код и данные на этом шаге не менял.


### Acknowledgments

| Работа | Подход в статье | Возможное применение |
| ------ | --------------- | -------------------- |
| [PPTOD, ACL 2022](https://aclanthology.org/2022.acl-long.319/) | T5; смешанное обучение задачам диалога с отдельными инструкциями | Организация смеси задач; состав связанного батча требует отдельной проверки |
| [SimpleTOD, 2020](https://arxiv.org/abs/2005.00796) | Одна causal LM; последовательность из контекста, состояния, результата запроса к БД, действия и ответа | Альтернатива с последовательной генерацией промежуточных результатов |
| [UBAR, AAAI 2021](https://arxiv.org/abs/2012.03539) | User-BeliefState-Act-Response (UBAR) on GPT-2; обучение на полных диалоговых сессиях и оценка с собственным сгенерированным контекстом | Проверка всей цепочки на предсказаниях модели. https://github.com/TonyNemo/UBAR-MultiWOZ |
| [SOM-DST, ACL 2020](https://aclanthology.org/2020.acl-main.53/) | Предсказание операций над слотами и генерация обновляемых значений | Упрощение цели `extraction`; вариант `observation` + `observe()` является нашей адаптацией |
