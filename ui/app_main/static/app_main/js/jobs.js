(() => {
  const monitor = document.getElementById('job-monitor');
  if (!monitor) return;
  const element = (id) => document.getElementById(id);
  const display = (id, visible) => { element(id).hidden = !visible; };
  const text = (id, value) => { element(id).textContent = value; };
  text('job-config', JSON.stringify(JSON.parse(element('job-config-data').textContent), null, 2));
  function render(data) {
    text('job-status', data.label);
    text('job-phase', data.phase);
    text('job-percent', `${Number(data.percent).toFixed(1)}%`);
    element('job-progress').value = data.percent;
    const p = data.progress;
    text('job-position', p.epoch !== undefined ? `Эпоха ${Number(p.epoch).toFixed(3)} / ${p.epochs} · шаг ${p.step || 0}` :
      p.total_episodes ? `Эпизоды ${p.completed_episodes} / ${p.total_episodes}${p.episode_id ? ` · ${p.episode_id}, ход ${p.turn}` : ''}` : '');
    text('job-loss', ['loss', 'eval_loss', 'learning_rate'].filter(key => p[key] !== undefined).map(key => `${key}: ${p[key]}`).join(' · '));
    text('job-error', data.error);
    display('job-error', Boolean(data.error));
    display('worker-warning', data.active && !data.worker_online);
    display('stop-form', data.can_stop);
    display('resume-link', data.resumable);
    display('result-link', Boolean(data.result_url));
    element('result-link').href = data.result_url;
    const log = element('job-log');
    const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 50;
    text('job-log', data.log || 'Задача ещё не начала выполнение.');
    if (atBottom) log.scrollTop = log.scrollHeight;
    display('dataset-stats', Boolean(p.dataset));
    if (p.dataset) text('dataset-values', JSON.stringify({train: p.dataset.train, validation: p.dataset.validation}, null, 2));
    return data.active;
  }
  async function poll() {
    let active = true;
    try {
      const response = await fetch(monitor.dataset.statusUrl, {cache: 'no-store'});
      if (!response.ok) throw new Error(response.status);
      active = render(await response.json());
      display('poll-warning', false);
    } catch (_) { display('poll-warning', true); }
    if (active) window.setTimeout(poll, 2000);
  }
  if (render(JSON.parse(element('job-initial-data').textContent))) window.setTimeout(poll, 2000);
})();
