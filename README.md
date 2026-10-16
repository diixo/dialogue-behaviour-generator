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
Chat with trained UBAR or History models through the same worker.
`UbarChat` predicts state → action → reply from the last state/action and a new user message;
`HistoryChat` generates a reply from conversation history.

CLI: `python chat_runtime.py --model runs/history` (or a UBAR checkpoint).
Commands: `/reset`, `/exit`; use `--device cuda` for GPU. Only `torch` and `transformers`
are required. To distribute the chat separately, copy `chat_runtime.py`, `inference.py`,
`dialogue_protocol.py`, `dialogue_format.py`, `requirements-chat.txt` and the model directory
(including tokenizer files and `experiment.json`).


## Data

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

Training tasks:

Во всех режимах вход включает инструкцию и текущее сообщение пользователя.

| **Task** | **Input** | **Output** | **Status** |
| -------- | --------- | ---------- | ---------- |
| `dst`    | Переписка + состояние до текущего сообщения пользователя | Полное обновлённое состояние (JSON) | Обучение и бенчмарк реализованы |
| `extraction` | Переписка + состояние до текущего сообщения пользователя | Наблюдение и полное обновлённое состояние (JSON) | Обучение и бенчмарк реализованы |
| `policy` | Переписка + обновлённое состояние | Следующее действие, например `ASK(mood)` | Обучение и бенчмарк реализованы |
| `response` | Переписка + обновлённое состояние + действие | Текст ответа | Обучение и бенчмарк реализованы |
| `history` | Переписка | Текст ответа | Обучение, бенчмарк и чат реализованы |
| `ubar`    | Последнее состояние и действие + новое сообщение пользователя | Состояние → действие → ответ | Обучение, бенчмарк и чат реализованы |

<details>
<summary>Шаблоны входа и выхода для каждого task (формат plain)</summary>

В первых пяти шаблонах вход заканчивается на последнем `assistant: `; далее
показана обучаемая цель. `<EOS>` обозначает нативный токен завершения, добавляемый
при токенизации, а не строку для записи в JSON. Формат выбирается через `--style`.

| Подстановка | Содержание |
| --- | --- |
| `{persona}` | Общая инструкция поведения `PERSONA` из [dialogue_protocol.py](dialogue_protocol.py) |
| `{history_messages}` | Предыдущая переписка в виде `user: ...` / `assistant: ...`, каждый элемент с переводом строки; на первом ходе пустая |
| `{user_message}` | Текущее сообщение пользователя; в `{history_messages}` его ещё нет |
| `{state_before_json}` | Полное состояние до текущего сообщения пользователя, уже учитывающее предыдущее действие ассистента; на первом ходе пустой снимок |
| `{state_json}` | Полное обновлённое состояние после текущего сообщения пользователя, до ответа ассистента |
| `{observation_json}` | Объект с полями `reaction`, `facts`, `new_topic`, `refused`, `user_question`, `exit` |
| `{action}` | Действие в строковом формате, например `ASK(mood)` или `CLOSE` |
| `{response}` | Текст следующей реплики ассистента |
| `{previous_state_json}` | Для UBAR: последнее предсказанное состояние до предыдущего действия ассистента; на первом ходе пустой снимок |
| `{previous_action}` | Для UBAR: последнее выбранное действие, например `ASK(activity)`; на первом ходе `NONE` |

**`dst` — только состояние.**

```text
system: Update the dialogue state using the latest user message and conversation. Return only the complete state as a JSON object. Preserve previous facts and refusals; do not invent unobserved information. State before the latest user message: {state_before_json}
{history_messages}user: {user_message}
assistant: {state_json}<EOS>
```

**`extraction` — наблюдение и состояние.**

```text
system: Read the latest user message and update the supplied dialogue state. Return a JSON object with observation and state. The observation contains reaction, facts, new_topic, refused, user_question and exit. The state contains all fields of the supplied state with accumulated known facts, slot statuses, question attempts, topic, used follow-ups and dialogue flags. Preserve previous facts and refusals; do not invent unobserved information.
State before the latest user message: {state_before_json}
{history_messages}user: {user_message}
assistant: {"observation": {observation_json}, "state": {state_json}}<EOS>
```

**`policy` — действие по обновлённому состоянию.**

```text
system: {persona} Predict the next dialogue action using the state and conversation.
Current dialogue state: {state_json}
{history_messages}user: {user_message}
assistant: {action}<EOS>
```

**`response` — ответ по обновлённому состоянию и заданному действию.**

```text
system: {persona} Write the next assistant message for this dialogue action: {action}.
Current dialogue state: {state_json}
{history_messages}user: {user_message}
assistant: {response}<EOS>
```

**`history` — прямой ответ по переписке.**

```text
system: {persona}
{history_messages}user: {user_message}
assistant: {response}<EOS>
```

**`ubar` — последовательность состояния, действия и ответа.**

```text
system: {persona} Track dialogue state, choose an action, and write a response in that order. The previous state was saved before the last assistant action. Account for that action once, then process the new user message. Preserve known facts and refusals; do not invent unobserved information.
Previous state: {previous_state_json}
Last assistant action: {previous_action}
user: {user_message}
<|state|>{state_json}<|end_state|><|action|>{action}<|end_action|><|response|>{response}<|end_response|><EOS>
```

В UBAR обучаемая цель каждого хода — три сегмента после пользовательского
сообщения. В чате они генерируются последовательно, до соответствующего
`<|end_...|>`; префикс `assistant: ` не используется. Нативный EOS добавляется
в конце каждого обучающего примера — одного хода; разговор завершается по `CLOSE`.
Маркеры UBAR добавляются при сборке последовательности, а не в исходные JSON/JSONL.
Старые реплики и снимки состояния модели не передаются. Новый формат отмечен
как `ubar_context: memory_v1` в `experiment.json`; прежние UBAR-модели нужно переобучить.

На обучении состояния и действия берутся из разметки; при оценке всей цепочки
и в чате — из предсказаний. Полная схема состояния и границы памяти описаны
в [FAQ.md](FAQ.md).

```text
Вход:
  Последнее состояние памяти
  Новое сообщение пользователя

Выход:
  Обновлённое состояние
  Выбранное действие
  Ответ пользователю
```

</details>

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
