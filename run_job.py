"""Execute one UI job in an isolated process; the web server never imports torch."""

import json
from pathlib import Path
import sys
import traceback

from job_runtime import FileLock, JobCancelled, JobControl, atomic_json


def execute(spec, control):
    config = spec['config']
    control.check_stop()
    if spec['kind'] == 'chat':
        from chat_runtime import generate_chat_turn
        from inference import HFPredictor
        control.emit(phase='loading', percent=0)
        predictor = HFPredictor(config['model'], device=config['device'], local_files_only=True,
                                max_new_tokens=config['max_new_tokens'])
        return generate_chat_turn(predictor, config['turns'], config['message'], control)
    from data import corpus_hash, read_episodes
    control.emit(phase='data', percent=0)
    episodes = {}
    for split, expected in config['data_hashes'].items():
        rows = read_episodes(config['data'], split)
        if corpus_hash(rows) != expected:
            raise ValueError(f'{split}: датасет изменился после постановки в очередь. Создайте новый запуск.')
        episodes[split] = rows[:config.get('limit_episodes')]
    control.check_stop()
    control.emit(phase='loading')
    if spec['kind'] == 'train':
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from trainer import train_experiment
        set_seed(config['seed'])
        source = config.get('resume') or config['model']
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=config['local_files_only'])
        model = AutoModelForCausalLM.from_pretrained(source, local_files_only=config['local_files_only'])
        control.check_stop()
        options = {key: config[key] for key in ('task', 'style', 'loss_scope', 'max_length', 'epochs',
                   'batch_size', 'accumulation', 'learning_rate', 'warmup_ratio', 'seed')}
        if config.get('resume'):
            # Resume into a new artifact directory, preserving the original experiment.
            manifest = json.loads((Path(config['resume']).parent / 'experiment.json').read_text(encoding='utf-8'))
            target = Path(spec['output'])
            if target.exists() and any(target.iterdir()):
                raise ValueError('Resume output must be a new directory')
            atomic_json(target / 'experiment.json', manifest)
        metrics = train_experiment(model, tokenizer, episodes['train'], episodes['validation'],
                                  output=spec['output'], model_name=config['model'],
                                  use_cpu=config['cpu'], resume=config.get('resume'),
                                  job_control=control, **options)
        return {'status': 'stopped' if metrics.get('stopped') else 'completed', 'metrics': metrics,
                'checkpoint': metrics.get('checkpoint')}
    if spec['kind'] != 'benchmark':
        raise ValueError('Unknown job kind')
    from benchmark import run_benchmark
    from generator import write_jsonl
    from inference import PredictorRouter
    output = Path(spec['output'])
    if output.exists() and any(output.iterdir()):
        raise ValueError('Benchmark output must be empty')
    predictor = PredictorRouter(config.get('model'), dst=config.get('dst'), policy=config.get('policy'),
                                response=config.get('response'), device=config['device'],
                                local_files_only=config['local_files_only'], max_new_tokens=config['max_new_tokens'])
    predictor.validate_mode(config['mode'])
    report, records = run_benchmark(episodes[config['split']], predictor, mode=config['mode'],
                                   seed=config['seed'], max_turns=config['max_turns'],
                                   progress=control.benchmark_progress)
    control.check_stop()
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / 'predictions.jsonl', records)
    atomic_json(output / 'report.json', report)
    return {'status': 'completed'}


def main(spec_path):
    spec_path = Path(spec_path)
    control = JobControl(spec_path.parent)
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    # Held by the child so a worker restart cannot overlap an orphaned GPU process.
    with FileLock(spec_path.parent.parent / 'execution.lock'):
        try:
            result = execute(spec, control)
        except JobCancelled as error:
            result = {'status': 'cancelled', 'error': str(error)}
        except Exception as error:
            traceback.print_exc()
            result = {'status': 'failed', 'error': str(error)}
        atomic_json(spec_path.parent / 'result.json', result)
    return 1 if result['status'] == 'failed' else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
