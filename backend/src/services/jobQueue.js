// Minimal in-process job queue. No Redis/Celery: a single-node deployment runs
// jobs sequentially and records state in MongoDB (see docs/decisions.md #4).
export function createJobQueue({ concurrency = 1, logger = null } = {}) {
  const pending = [];
  let active = 0;
  let completed = 0;
  let failed = 0;

  function pump() {
    while (active < concurrency && pending.length > 0) {
      const job = pending.shift();
      active += 1;
      Promise.resolve()
        .then(job.run)
        .then(() => {
          completed += 1;
        })
        .catch((err) => {
          failed += 1;
          logger?.error?.("job failed", {
            job: job.name,
            message: err?.message || String(err),
          });
        })
        .finally(() => {
          active -= 1;
          pump();
        });
    }
  }

  function enqueue(name, run) {
    pending.push({ name, run });
    pump();
    return { name, queued: pending.length, active };
  }

  function stats() {
    return { concurrency, active, pending: pending.length, completed, failed };
  }

  return { enqueue, stats };
}
