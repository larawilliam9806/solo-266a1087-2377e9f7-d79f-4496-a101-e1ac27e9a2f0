"""Storage cleanup: reclaim blobs left by abandoned previews and crashed publishes."""

import hashlib
import threading

import pytest
from conftest import DEFAULT_CSV, preview_and_confirm, upload

from railapp import create_app, db


def _orphan(store, data):
    """Write a blob with no database reference, like a crashed publish would."""
    return store.put_bytes(data)


def _preview_digests(payload):
    digests = {payload["original"]["sha256"]}
    for outcome in payload["downstreams"].values():
        if outcome["output_sha256"]:
            digests.add(outcome["output_sha256"])
    return digests


def test_cleanup_deletes_orphans_and_reports_counts(client, app):
    _, confirmed = preview_and_confirm(client)
    assert confirmed.status_code == 201
    store = app.config["STORE"]
    published = len(store.iter_blobs())
    assert published == 3

    orphan_a = _orphan(store, b"orphan,a\n1,2\n")
    orphan_b = _orphan(store, b"another interrupted publish blob")
    size_a = len(b"orphan,a\n1,2\n")
    size_b = len(b"another interrupted publish blob")

    response = client.post("/api/cleanup")
    assert response.status_code == 200
    report = response.get_json()
    assert report["scanned"] == published + 2
    assert report["kept"] == published
    assert report["deleted"] == 2
    assert report["freed_bytes"] == size_a + size_b
    assert report["failed"] == []
    assert not store.exists(orphan_a)
    assert not store.exists(orphan_b)

    # Published revision is untouched and still downloadable.
    for kind in ("original", "settlement", "ridership"):
        assert client.get(f"/download/2026-11-01/1/{kind}").status_code == 200
    assert len(store.iter_blobs()) == published


def test_cleanup_keeps_active_preview_blobs(client, app):
    payload = upload(client).get_json()
    digests = _preview_digests(payload)
    store = app.config["STORE"]
    assert len(store.iter_blobs()) == len(digests) == 3

    report = client.post("/api/cleanup").get_json()
    assert report["deleted"] == 0
    assert report["kept"] == 3
    for digest in digests:
        assert store.exists(digest)


def test_cleanup_reclaims_discarded_preview(client, app):
    payload = upload(client).get_json()
    digests = _preview_digests(payload)
    store = app.config["STORE"]

    discarded = client.delete(f"/api/previews/{payload['preview_id']}")
    assert discarded.status_code == 200

    report = client.post("/api/cleanup").get_json()
    assert report["deleted"] == 3
    for digest in digests:
        assert not store.exists(digest)


def test_cleanup_reclaims_expired_preview_but_keeps_active(client, app):
    store = app.config["STORE"]
    stale = upload(client, day="2026-11-01").get_json()
    # Change a ridership-visible column too, so the two previews share no
    # content-addressed blob at all.
    fresh = upload(client, day="2026-11-02", data=DEFAULT_CSV.replace("G22", "G23").encode()).get_json()
    stale_digests = _preview_digests(stale)
    fresh_digests = _preview_digests(fresh)

    conn = db.connect(app.config["DB_PATH"])
    try:
        conn.execute(
            "UPDATE previews SET expires_at = '2000-01-01T00:00:00Z' WHERE id = ?",
            (stale["preview_id"],),
        )
    finally:
        conn.close()

    report = client.post("/api/cleanup").get_json()
    assert report["deleted"] == len(stale_digests)
    for digest in stale_digests:
        assert not store.exists(digest)
    for digest in fresh_digests:
        assert store.exists(digest)


def test_cleanup_keeps_every_published_revision(client, app):
    preview_and_confirm(client)
    preview_and_confirm(client, data=DEFAULT_CSV.replace("G22,88", "G22,99").encode())
    store = app.config["STORE"]
    before = {b["digest"] for b in store.iter_blobs()}

    report = client.post("/api/cleanup").get_json()
    assert report["deleted"] == 0
    assert {b["digest"] for b in store.iter_blobs()} == before
    for rev in (1, 2):
        for kind in ("original", "settlement", "ridership"):
            assert client.get(f"/download/2026-11-01/{rev}/{kind}").status_code == 200


def test_cleanup_skips_blob_referenced_after_scan(client, app, monkeypatch):
    """A preview created between the scan and the delete keeps its blob."""
    store = app.config["STORE"]
    raw = DEFAULT_CSV.encode()
    orphan_digest = hashlib.sha256(raw).hexdigest()
    store.put_bytes(raw)  # orphan for now: no preview row references it yet
    cleaner = app.extensions["cleaner"]

    real_refs = db.reference_digests
    calls = {"n": 0}

    def refs_then_upload(conn, now=None):
        calls["n"] += 1
        if calls["n"] == 2:
            # First per-candidate re-check: the upload happens *after* the
            # scan but *before* this file's delete decision.
            assert upload(client, day="2026-11-06", data=raw).status_code == 201
        return real_refs(conn, now)

    monkeypatch.setattr(db, "reference_digests", refs_then_upload)
    report = cleaner.run()
    assert calls["n"] >= 2
    assert report["skipped"] == 1
    assert report["deleted"] == 0
    assert store.exists(orphan_digest)


def test_inflight_registry_protects_unregistered_blob(app):
    """A digest in use by an in-progress operation is kept even with no row."""
    store = app.config["STORE"]
    cleaner = app.extensions["cleaner"]
    digest = _orphan(store, b"not yet registered anywhere")

    with cleaner.registry.guard([digest]):
        report = cleaner.run()
        assert report["deleted"] == 0
        assert store.exists(digest)

    followup = cleaner.run()
    assert followup["deleted"] == 1
    assert not store.exists(digest)


def test_cleanup_during_inflight_preview_keeps_its_blobs(client, app, monkeypatch):
    """Cleanup running mid-preview must not reclaim the half-written upload."""
    service = app.extensions["service"]
    store = app.config["STORE"]
    cleaner = app.extensions["cleaner"]
    raw = DEFAULT_CSV.encode()
    digest = hashlib.sha256(raw).hexdigest()

    blob_written = threading.Event()
    finish_preview = threading.Event()
    real_put = store.put_bytes

    def pausing_put(data):
        result = real_put(data)
        if hashlib.sha256(data).hexdigest() == digest:
            # Original blob is on disk, the preview row does not exist yet.
            blob_written.set()
            assert finish_preview.wait(10)
        return result

    monkeypatch.setattr(store, "put_bytes", pausing_put)
    outcome = {}

    def run_preview():
        outcome["result"] = service.preview("2026-11-07", "tickets.csv", raw)

    thread = threading.Thread(target=run_preview)
    thread.start()
    try:
        assert blob_written.wait(10)
        report = cleaner.run()
        assert store.exists(digest)
        assert report["deleted"] == 0
    finally:
        finish_preview.set()
        thread.join(10)
    status, payload = outcome["result"]
    assert status == 201
    assert payload["original"]["sha256"] == digest


def test_preview_and_confirm_do_not_wait_for_cleanup_round(client, app, monkeypatch):
    """Upload preview and confirm publish proceed while a round is paused."""
    store = app.config["STORE"]
    cleaner = app.extensions["cleaner"]
    for i in range(3):
        _orphan(store, f"orphan-{i}".encode())

    real_refs = db.reference_digests
    calls = {"n": 0}
    round_paused = threading.Event()
    allow_finish = threading.Event()

    def pausing_refs(conn, now=None):
        calls["n"] += 1
        if calls["n"] == 2:
            # The round is paused mid-delete-pass, holding no locks.
            round_paused.set()
            assert allow_finish.wait(10)
        return real_refs(conn, now)

    monkeypatch.setattr(db, "reference_digests", pausing_refs)
    result = {}

    def run_cleanup():
        result["report"] = cleaner.run()

    thread = threading.Thread(target=run_cleanup)
    thread.start()
    try:
        assert round_paused.wait(10)
        # If preview/confirm had to wait for the round, this would deadlock:
        # the round only finishes once allow_finish is set below.
        preview, confirmed = preview_and_confirm(
            client, day="2026-11-08", data=DEFAULT_CSV.replace("T1", "T8").encode()
        )
        assert confirmed.status_code == 201
    finally:
        allow_finish.set()
        thread.join(10)
    report = result["report"]
    assert report["deleted"] == 3
    assert report["failed"] == []
    # The publication committed during the round keeps its blobs.
    for kind in ("original", "settlement", "ridership"):
        assert client.get(f"/download/2026-11-08/1/{kind}").status_code == 200


def test_cleanup_during_inflight_confirm_keeps_everything_downloadable(client, app, monkeypatch):
    service = app.extensions["service"]
    cleaner = app.extensions["cleaner"]
    preview = upload(client, day="2026-11-10").get_json()

    confirm_started = threading.Event()
    finish_confirm = threading.Event()
    real_parse = __import__("railapp.convert", fromlist=["parse_csv_bytes"]).parse_csv_bytes

    def pausing_parse(raw):
        confirm_started.set()
        assert finish_confirm.wait(10)
        return real_parse(raw)

    monkeypatch.setattr("railapp.publisher.convert.parse_csv_bytes", pausing_parse)
    outcome = {}

    def run_confirm():
        outcome["result"] = service.confirm(preview["preview_id"])

    thread = threading.Thread(target=run_confirm)
    thread.start()
    try:
        assert confirm_started.wait(10)
        # Confirm holds the write transaction and has registered its digests;
        # a cleanup round now must not delete anything it needs.
        report = cleaner.run()
        assert report["deleted"] == 0
    finally:
        finish_confirm.set()
        thread.join(10)
    status, payload = outcome["result"]
    assert status == 201
    for kind in ("original", "settlement", "ridership"):
        assert client.get(f"/download/2026-11-10/1/{kind}").status_code == 200


def test_failed_deletions_are_reported_and_retriable(client, app, monkeypatch):
    store = app.config["STORE"]
    cleaner = app.extensions["cleaner"]
    good = _orphan(store, b"delete me")
    bad = _orphan(store, b"cannot delete me")

    real_delete = store.delete

    def flaky_delete(digest):
        if digest == bad:
            raise OSError("simulated I/O error")
        return real_delete(digest)

    monkeypatch.setattr(store, "delete", flaky_delete)
    report = cleaner.run()
    assert report["deleted"] == 1
    assert report["freed_bytes"] == len(b"delete me")
    assert len(report["failed"]) == 1
    failure = report["failed"][0]
    assert failure["digest"] == bad
    assert failure["size"] == len(b"cannot delete me")
    assert "simulated I/O error" in failure["error"]
    assert store.exists(bad)
    assert not store.exists(good)

    # The failure is visible through the API and a retry cleans it up.
    status = client.get("/api/cleanup").get_json()
    assert status["last_report"]["failed"][0]["digest"] == bad
    monkeypatch.setattr(store, "delete", real_delete)
    retry = client.post("/api/cleanup").get_json()
    assert retry["deleted"] == 1
    assert retry["failed"] == []
    assert not store.exists(bad)


def test_cleanup_after_restart_reclaims_interrupted_leftovers(tmp_path):
    """A fresh process over the same data dir cleans what a crash left."""
    data_dir = str(tmp_path / "data")
    first = create_app({"DATA_DIR": data_dir, "TESTING": True})
    client1 = first.test_client()
    payload = upload(client1).get_json()
    assert client1.delete(f"/api/previews/{payload['preview_id']}").status_code == 200
    store1 = first.config["STORE"]
    store1.put_bytes(b"blob from a publish interrupted by a crash")
    assert len(store1.iter_blobs()) == 4

    # Restart: brand new app instance, same DATA_DIR.
    second = create_app({"DATA_DIR": data_dir, "TESTING": True})
    client2 = second.test_client()
    report = client2.post("/api/cleanup").get_json()
    assert report["deleted"] == 4
    assert report["failed"] == []
    assert second.config["STORE"].iter_blobs() == []


def test_cleanup_ignores_unrecognised_files(client, app):
    store = app.config["STORE"]
    junk_outside_shard = store.objects + "/notes.txt"
    with open(junk_outside_shard, "w") as handle:
        handle.write("operator notes")
    shard = store.objects + "/zz"
    import os

    os.makedirs(shard, exist_ok=True)
    with open(shard + "/not-a-blob", "w") as handle:
        handle.write("junk")
    orphan = _orphan(store, b"real orphan")

    report = client.post("/api/cleanup").get_json()
    assert report["deleted"] == 1
    assert not store.exists(orphan)
    with open(junk_outside_shard) as handle:
        assert handle.read() == "operator notes"
    with open(shard + "/not-a-blob") as handle:
        assert handle.read() == "junk"


def test_concurrent_cleanup_rounds_conflict(client, app):
    cleaner = app.extensions["cleaner"]
    assert cleaner._run_lock.acquire(blocking=False)
    try:
        response = client.post("/api/cleanup")
        assert response.status_code == 409
        assert response.get_json()["error"] == "cleanup_in_progress"
    finally:
        cleaner._run_lock.release()
    assert client.post("/api/cleanup").status_code == 200


def test_cleanup_status_endpoint_reports_store(client, app):
    _orphan(app.config["STORE"], b"floating garbage")
    status = client.get("/api/cleanup").get_json()
    assert status["running"] is False
    assert status["store"]["objects"] == 1
    assert status["store"]["unreferenced"] == 1
    assert status["store"]["unreferenced_bytes"] == len(b"floating garbage")
    assert status["last_report"] is None

    client.post("/api/cleanup")
    status = client.get("/api/cleanup").get_json()
    assert status["store"]["objects"] == 0
    assert status["last_report"]["deleted"] == 1


def test_cleanup_page_and_form(client, app):
    _orphan(app.config["STORE"], b"page orphan")
    page = client.get("/cleanup")
    assert page.status_code == 200
    assert "存储清理" in page.get_data(as_text=True)
    assert "可回收" in page.get_data(as_text=True)

    run = client.post("/cleanup/run")
    assert run.status_code == 302
    assert run.headers["Location"].endswith("/cleanup?ran=1")
    page = client.get("/cleanup?ran=1")
    text = page.get_data(as_text=True)
    assert "实际删除文件数" in text
    assert "释放空间" in text

    # Navigation links to the page.
    index = client.get("/").get_data(as_text=True)
    assert "/cleanup" in index


def test_cleanup_page_shows_failures(client, app, monkeypatch):
    store = app.config["STORE"]
    cleaner = app.extensions["cleaner"]
    bad = _orphan(store, b"stuck blob")

    def always_fail(digest):
        raise OSError("disk is read-only")

    monkeypatch.setattr(store, "delete", always_fail)
    cleaner.run()
    text = client.get("/cleanup").get_data(as_text=True)
    assert bad in text
    assert "disk is read-only" in text
    assert "再次运行清理即可重试" in text


def test_preview_and_confirm_register_inflight_digests(client, app, monkeypatch):
    service = app.extensions["service"]
    seen = []
    real_guard = service.registry.guard

    class Recorder:
        def guard(self, digests):
            seen.append(set(digests))
            return real_guard(digests)

    monkeypatch.setattr(service, "registry", Recorder())

    payload = upload(client, day="2026-11-12").get_json()
    expected = _preview_digests(payload)
    assert seen and seen[0] == expected

    confirmed = client.post(
        f"/api/previews/{payload['preview_id']}/confirm",
        headers={"Accept": "application/json"},
    )
    assert confirmed.status_code == 201
    assert seen[1] == expected
