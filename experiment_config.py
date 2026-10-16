"""Lightweight defaults shared by CLI training and the local dashboard."""

TRAIN_TASKS = ('dst', 'extraction', 'policy', 'response', 'history', 'ubar')
EVAL_CHAINS = ('dst', 'extraction', 'policy', 'response', 'history', 'policy_response', 'full')
EVAL_CONTEXTS = ('reference', 'predicted')
EVAL_USER_SOURCES = ('dataset', 'simulator')
TRAIN_DEFAULTS = dict(model='gpt2', task='ubar', style='plain', loss_scope='targets',
                      max_length=1024, epochs=2, batch_size=4, accumulation=1,
                      learning_rate=8e-5, warmup_ratio=0.05, seed=42)


def validate_evaluation(chain, context, user_source):
    if chain not in EVAL_CHAINS or context not in EVAL_CONTEXTS or user_source not in EVAL_USER_SOURCES:
        raise ValueError('Unknown evaluation chain, context or user source')
    if context == 'predicted' and chain != 'full':
        raise ValueError('Predicted context requires the full DST -> policy -> response chain')
    if user_source == 'simulator' and context != 'predicted':
        raise ValueError('Simulated users require predicted context')


def evaluation_options(config):
    """Read current options or translate a historical report/queued job at the boundary."""
    if 'chain' in config or 'user_source' in config:
        options = {key: config.get(key) for key in ('chain', 'context', 'user_source')}
    else:
        mode = config.get('mode')
        options = dict(chain='full' if mode in ('session', 'rollout') else mode,
                       context='predicted' if mode in ('session', 'rollout') else 'reference',
                       user_source='simulator' if mode == 'rollout' else 'dataset')
    validate_evaluation(**options)
    return options


def required_stages(chain):
    return {'full': ('dst', 'policy', 'response'),
            'policy_response': ('policy', 'response')}.get(chain, (chain,))


def compatible_task(trained, stage):
    return (trained == stage or (stage == 'dst' and trained == 'extraction') or
            (trained == 'ubar' and stage in ('dst', 'policy', 'response')))
