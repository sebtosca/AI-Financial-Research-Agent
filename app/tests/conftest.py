import os


# Unit tests must not emit external traces from a developer's local .env file.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
