.PHONY: proto test ingestor import-duckdb backfill-gap lint fmt

PROTO_SRC = proto/ingestor/v1/ingestor.proto

proto:
	uv run python -m grpc_tools.protoc \
		-I . \
		--python_out=. \
		--grpc_python_out=. \
		--pyi_out=. \
		$(PROTO_SRC)

ingestor:
	uv run python -m ingestor

import-duckdb:
	uv run python -m scripts.import_duckdb $(ARGS)

backfill-gap:
	uv run python -m scripts.backfill_gap $(ARGS)

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .

fmt:
	uv run ruff format .
