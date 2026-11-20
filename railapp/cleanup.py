"""Reclaim blob files that nothing references any more.

Abandoned previews (discarded or expired) and interrupted publishes leave
content-addressed blobs on disk that no database row points at.  The cleaner
in this module scans the blob store, computes the live reference set, and
deletes the rest — carefully, because uploads and confirms keep running:

* References are re-checked *per file, immediately before deleting it*, so a
  blob that became referenced by a preview or publication after the scan is
  skipped instead of deleted.
* Preview and confirm register the digests they are using in an in-memory
  :class:`InFlightRegistry` before writing any blob and keep them registered
  until the database row that protects them commits.  The cleaner treats
  registered digests as referenced even though no row mentions them yet.
* The registry check and the ``unlink`` happen under the same mutex
  acquisition, so a writer cannot slip a reference in between the check and
  the delete.  The critical section is a single ``unlink`` — microseconds —
  so uploads and confirms never wait for a cleanup round, only (rarely) for
  one file's deletion.
* The cleaner never takes a database write lock; its queries are short
  autocommit reads, which under WAL never block writers.

All state is derived from the database and the filesystem, so a round started
after a restart reclaims exactly what an interrupted earlier process left
behind.
"""

import threading
from collections import Counter
from contextlib import contextmanager

from . import db
from .errors import Conflict


class InFlightRegistry:
    """Digests currently in use by an in-progress preview or confirm.

    Preview writes blobs before the preview row commits, and confirm may
    re-write blobs before the publication row commits.  In those windows no
    database row references the blob yet, so the cleaner must consult this
    registry in addition to the database.  The deployment is single-process
    (one gunicorn worker), so an in-memory registry is authoritative.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._counts = Counter()

    @contextmanager
    def guard(self, digests):
        """Keep ``digests`` protected for the duration of the block."""
        self._register(digests)
        try:
            yield
        finally:
            self._deregister(digests)

    def _register(self, digests):
        with self._lock:
            for digest in digests:
                self._counts[digest] += 1

    def _deregister(self, digests):
        with self._lock:
            for digest in digests:
                remaining = self._counts.get(digest, 0)
                if remaining <= 1:
                    self._counts.pop(digest, None)
                else:
                    self._counts[digest] = remaining - 1

    def __contains__(self, digest):
        with self._lock:
            return digest in self._counts

    def snapshot(self):
        with self._lock:
            return set(self._counts)

    @contextmanager
    def locked(self):
        """Hold the registry mutex across a check-and-delete critical section."""
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()


class BlobCleaner:
    """Scans the blob store and deletes files nothing references."""

    def __init__(self, conn_factory, store, registry):
        self.conn_factory = conn_factory
        self.store = store
        self.registry = registry
        self._run_lock = threading.Lock()
        self._last_report = None

    @property
    def running(self):
        return self._run_lock.locked()

    @property
    def last_report(self):
        return self._last_report

    def summary(self):
        """Current store usage and how much of it is reclaimable right now."""
        blobs = self.store.iter_blobs()
        conn = self.conn_factory()
        try:
            referenced = db.reference_digests(conn)
        finally:
            conn.close()
        protected = referenced | self.registry.snapshot()
        garbage = [b for b in blobs if b["digest"] not in protected]
        return {
            "objects": len(blobs),
            "bytes": sum(b["size"] for b in blobs),
            "referenced": len(blobs) - len(garbage),
            "unreferenced": len(garbage),
            "unreferenced_bytes": sum(b["size"] for b in garbage),
        }

    def run(self):
        """Run one cleanup round and return its report.

        Only one round may run at a time; a concurrent request gets 409.
        The round never blocks previews or confirms: it holds no database
        write lock and its per-file critical section is a single unlink.
        """
        if not self._run_lock.acquire(blocking=False):
            raise Conflict(
                "上一轮清理仍在进行中，请稍后查看结果", code="cleanup_in_progress"
            )
        try:
            report = self._run_round()
            self._last_report = report
            return report
        finally:
            self._run_lock.release()

    def _run_round(self):
        started_at = db.iso(db.utcnow())
        blobs = self.store.iter_blobs()
        scanned = len(blobs)
        kept = 0
        deleted = 0
        freed_bytes = 0
        skipped = 0
        failed = []
        conn = self.conn_factory()
        try:
            protected = db.reference_digests(conn) | self.registry.snapshot()
            candidates = [b for b in blobs if b["digest"] not in protected]
            kept = scanned - len(candidates)
            for blob in candidates:
                digest = blob["digest"]
                # Re-check with fresh state: anything referenced by a preview
                # or publication committed after the scan must be skipped.
                if (
                    digest in db.reference_digests(conn)
                    or digest in self.registry
                ):
                    skipped += 1
                    continue
                # The registry check and the unlink must be one atomic step:
                # writers register a digest before they can observe or commit
                # it, so holding the mutex across both closes the race.
                with self.registry.locked():
                    if digest in self.registry:
                        skipped += 1
                        continue
                    try:
                        size = self.store.delete(digest)
                    except OSError as exc:
                        failed.append(
                            {
                                "digest": digest,
                                "size": blob["size"],
                                "error": str(exc),
                            }
                        )
                        continue
                if size is None:
                    # Already gone (removed outside this round); nothing was
                    # actually deleted, so it is neither counted nor failed.
                    continue
                deleted += 1
                freed_bytes += size
        finally:
            conn.close()
        return {
            "started_at": started_at,
            "finished_at": db.iso(db.utcnow()),
            "scanned": scanned,
            "kept": kept,
            "deleted": deleted,
            "freed_bytes": freed_bytes,
            "skipped": skipped,
            "failed": failed,
        }
