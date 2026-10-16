(() => {
  const monitor = document.getElementById('chat-monitor');
  if (!monitor) return;
  const transcript = document.getElementById('chat-transcript');
  transcript.scrollTop = transcript.scrollHeight;
  const form = document.getElementById('chat-message-form');
  if (form) form.addEventListener('submit', () => {
    form.querySelector('button[type="submit"]').disabled = true;
  });
  async function poll() {
    try {
      const response = await fetch(monitor.dataset.statusUrl, {cache: 'no-store'});
      if (!response.ok) throw new Error(response.status);
      const data = await response.json();
      if (!data.active) { window.location.reload(); return; }
      document.getElementById('chat-phase').textContent = data.phase;
      document.getElementById('chat-worker-warning').hidden = data.worker_online;
      document.getElementById('chat-poll-warning').hidden = true;
    } catch (_) { document.getElementById('chat-poll-warning').hidden = false; }
    window.setTimeout(poll, 1500);
  }
  if (monitor.dataset.active === 'true') window.setTimeout(poll, 1500);
})();
