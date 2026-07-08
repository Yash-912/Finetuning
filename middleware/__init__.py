from .classifier import InjectionClassifier


def wrap_openai_client(client, endpoint="http://localhost:8080"):
    """Drop-in wrapper: redirects an OpenAI client through our middleware."""
    client.base_url = endpoint
    return client


__all__ = ["InjectionClassifier", "wrap_openai_client"]
