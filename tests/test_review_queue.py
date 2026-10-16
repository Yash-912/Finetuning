from scripts.review_queue import ReviewQueue


def test_ids_do_not_collide_across_restarts(tmp_path):
    db = tmp_path / "q.db"
    q1 = ReviewQueue(str(db))
    first = [q1.add(f"a{i}", 0, 0.9) for i in range(3)]
    q1.close()

    q2 = ReviewQueue(str(db))  # simulates a server restart
    second = q2.add("after restart", 1, 0.95)
    assert second not in first
    assert q2.stats()["total_requests"] == 4
