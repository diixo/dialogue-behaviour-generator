"""Lightweight defaults shared by CLI training and the local dashboard."""

TRAIN_TASKS = ('dst', 'extraction', 'policy', 'response', 'history', 'ubar')
EVAL_MODES = ('dst', 'extraction', 'policy', 'response', 'history', 'policy_response', 'full', 'rollout')
TRAIN_DEFAULTS = dict(model='gpt2', task='ubar', style='plain', loss_scope='targets',
                      max_length=1024, epochs=2, batch_size=4, accumulation=1,
                      learning_rate=8e-5, warmup_ratio=0.05, seed=42)


def required_stages(mode):
    return {'full': ('dst', 'policy', 'response'), 'rollout': ('dst', 'policy', 'response'),
            'policy_response': ('policy', 'response')}.get(mode, (mode,))


def compatible_task(trained, stage):
    return (trained == stage or (stage == 'dst' and trained == 'extraction') or
            (trained == 'ubar' and stage in ('dst', 'policy', 'response')))
