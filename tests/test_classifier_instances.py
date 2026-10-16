import middleware.classifier as c


def test_instances_are_distinct(monkeypatch):
    monkeypatch.setattr(c.InjectionClassifier, "_load_model", lambda self: None)
    monkeypatch.setattr(c.InjectionClassifier, "_warmup", lambda self: None)
    prod = c.InjectionClassifier(model_path="a")
    shadow = c.InjectionClassifier(model_path="b")
    assert prod is not shadow
    assert str(prod.model_path) == "a" and str(shadow.model_path) == "b"
    assert prod.is_ready and shadow.is_ready
