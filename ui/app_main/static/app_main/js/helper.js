
export function clientLog(tag, payload = {}) {
  try {
    const body = { ...payload, ts: Date.now() / 1000 };
    console.log('[client-log]', tag, body);
  } catch (e) {
    // eslint-disable-next-line no-console
    console.warn('[client-log] failed', tag, e);
  }
}

export function formatTime(sec) {
  if (!sec || sec <= 0) return '0s';
  const s = Math.floor(sec % 60);
  const m = Math.floor(sec / 60);
  if (m > 0) {
    return `${m}m ${s}s`;
  }
  return `${s}s`;
}

