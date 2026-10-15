"""Train independent task checkpoints or a UBAR session model.

Examples:
  python trainer.py --task dst --output runs/dst
  python trainer.py --task ubar --output runs/ubar
  python trainer.py --task policy --output runs/policy --max-steps 500

All modes use canonical episodes; labels never include hidden simulator facts.
The default pretrained model for every mode is the classic GPT-2 (gpt2).
Use benchmark.py for generation quality: validation loss selects checkpoints only.
"""

import argparse
from functools import partial
import json
from pathlib import Path

from data import TASKS, SPECIAL_TOKENS, collate_batch, corpus_hash, make_dataset, read_episodes
from dialogue_format import FORMATS


def dataset_stats(dataset):
    return {'examples': len(dataset),
            'input_tokens': sum(len(item['input_ids']) for item in dataset),
            'target_tokens': sum(int((item['labels'][1:] != -100).sum()) for item in dataset)}


def train_experiment(model, tokenizer, train_episodes, validation_episodes, *, task, output,
                     model_name='provided-model', style='plain', max_length=1024,
                     loss_scope='targets', epochs=3, max_steps=-1, batch_size=2,
                     accumulation=1, learning_rate=5e-5, warmup_ratio=0.05, seed=42,
                     use_cpu=False, resume=None):
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(seed)
    if not train_episodes or not validation_episodes:
        raise ValueError('Training and validation episodes must both be nonempty')
    if {ep['episode_id'] for ep in train_episodes} & {ep['episode_id'] for ep in validation_episodes}:
        raise ValueError('Training and validation episode IDs overlap')
    if epochs <= 0 or batch_size <= 0 or accumulation <= 0 or max_steps == 0 or max_steps < -1:
        raise ValueError('Invalid training budget or batch size')
    output = Path(output)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise ValueError(f'{output} is not empty; choose another output or --resume a checkpoint')
    if tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer requires an EOS token')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    if task == 'ubar':
        tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
        if len(tokenizer) != model.get_input_embeddings().num_embeddings:
            model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    context_limit = getattr(model.config, 'max_position_embeddings', None)
    if context_limit is not None and max_length > context_limit:
        raise ValueError(f'max_length={max_length} exceeds model context {context_limit}')
    options = dict(style=style, max_length=max_length, loss_scope=loss_scope)
    train_data = make_dataset(train_episodes, tokenizer, task, **options)
    validation_data = make_dataset(validation_episodes, tokenizer, task, **options)
    if not train_data or not validation_data:
        raise ValueError('Selected task has no examples')
    manifest = {'schema_version': 1, 'task': task, 'base_model': model_name,
                'style': style, 'max_length': max_length, 'loss_scope': loss_scope,
                'seed': seed, 'epochs': epochs, 'max_steps': max_steps, 'batch_size': batch_size,
                'gradient_accumulation_steps': accumulation, 'learning_rate': learning_rate,
                'warmup_ratio': warmup_ratio,
                'train_hash': corpus_hash(train_episodes),
                'validation_hash': corpus_hash(validation_episodes),
                'train': dataset_stats(train_data), 'validation': dataset_stats(validation_data)}
    config_path = output / 'experiment.json'
    if resume is not None:
        if not config_path.exists():
            raise ValueError('Resume requires experiment.json in the output directory')
        previous = json.loads(config_path.read_text(encoding='utf-8'))
        for key in manifest:
            if previous.get(key) != manifest[key]:
                raise ValueError(f'Resume configuration differs: {key}')
    output.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    args = TrainingArguments(
        output_dir=str(output), num_train_epochs=epochs, max_steps=max_steps,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accumulation, learning_rate=learning_rate,
        warmup_ratio=warmup_ratio, weight_decay=0.01, seed=seed, data_seed=seed,
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
        report_to=[], logging_steps=10, use_cpu=use_cpu, optim='adamw_torch',
        dataloader_pin_memory=not use_cpu,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_data, eval_dataset=validation_data,
                      processing_class=tokenizer,
                      data_collator=partial(collate_batch, padding_value=tokenizer.pad_token_id))
    result = trainer.train(resume_from_checkpoint=resume)
    metrics = {**result.metrics, **trainer.evaluate(), 'global_step': trainer.state.global_step,
               'best_checkpoint': trainer.state.best_model_checkpoint}
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--model', default='gpt2')
    parser.add_argument('--data', default='data', help='Directory containing canonical *.episodes.jsonl/json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--style', choices=FORMATS, default='plain')
    parser.add_argument('--loss-scope', choices=('targets', 'all'), default='targets', help='all is UBAR only')
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--epochs', type=float, default=3)
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit-episodes', type=int, help='Deterministic prefix of each split for smoke runs')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--resume', help='Checkpoint path; keep the original experiment arguments')
    args = parser.parse_args()
    if args.limit_episodes is not None and args.limit_episodes <= 0:
        parser.error('--limit-episodes must be positive')
    train = read_episodes(args.data, 'train')[:args.limit_episodes]
    validation = read_episodes(args.data, 'validation')[:args.limit_episodes]
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=args.local_files_only)
    metrics = train_experiment(
        model, tokenizer, train, validation, task=args.task, output=args.output, model_name=args.model,
        style=args.style, max_length=args.max_length, loss_scope=args.loss_scope,
        epochs=args.epochs, max_steps=args.max_steps, batch_size=args.batch_size,
        accumulation=args.accumulation, learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio, seed=args.seed, use_cpu=args.cpu, resume=args.resume)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
