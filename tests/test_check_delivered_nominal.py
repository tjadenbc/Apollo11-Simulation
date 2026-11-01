import json

import check_artifact as CA
import check_delivered_nominal as CDN


def test_artifact_clean(package_path):
    doc = CA.load(package_path)
    r = CA.check(doc, "clean")
    assert not r.fails, r.text()
    assert "A8.hash" not in r.fail_ids


def test_nominal_accept(package_path):
    doc = CA.load(package_path)
    r = CDN.check(doc, "clean", use_physics=True)
    assert not r.rejects(), r.text()
    assert r.verdict() in ("ACCEPT", "ACCEPT-WITH-WARNINGS")
    ids = {row[1] for row in r.rows}
    for must in ("T4.moon_ephemeris", "T4.ground_track", "T9.tli_coast", "T9.loi_coast", "T9.tei_coast", "T7.shell", "T5.retrograde"):
        assert must in ids, must


def test_self_test_every_check_fires(package_path):
    assert CDN.self_test(package_path, use_physics=True)


def test_self_test_without_physics(package_path):
    assert CDN.self_test(package_path, use_physics=False)
