"""RAG context injector — queries a ChromaDB vector store and silently
prepends relevant historical context to the prompt before it hits the AI."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import chromadb

from src.core.models import Message, Role
from src.core.state import PipelineState
from src.middleware.base import MiddlewareBase

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = Path.home() / ".ai-gateway" / "chroma_db"


class RAGInjectorMiddleware(MiddlewareBase):
    """Pre-processing middleware that injects relevant past context."""

    name = "rag_injector"

    def __init__(
        self,
        persist_dir: Path | str | None = None,
        collection_name: str = "conversations",
        n_results: int = 3,
        enabled: bool = True,
    ) -> None:
        self.persist_dir = Path(persist_dir) if persist_dir else DEFAULT_CHROMA_PATH
        self.collection_name = collection_name
        self.n_results = n_results
        self.enabled = enabled
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    def _ensure_collection(self) -> chromadb.Collection:
        if self._collection is None:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir)
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
            )
        return self._collection

    # ------------------------------------------------------------------
    # Ingest (call after getting a response to grow the brain)
    # ------------------------------------------------------------------
    def ingest(self, request_id: str, user_text: str, assistant_text: str) -> None:
        """Store a conversation turn in the vector DB for future retrieval."""
        collection = self._ensure_collection()
        doc = f"User: {user_text}\nAssistant: {assistant_text}"
        collection.add(
            documents=[doc],
            ids=[request_id],
            metadatas=[{"user": user_text[:200], "assistant": assistant_text[:200]}],
        )
        logger.debug("Ingested conversation %s into RAG", request_id)

    # ------------------------------------------------------------------
    # Pipeline hook
    # ------------------------------------------------------------------
    async def process(self, state: PipelineState) -> PipelineState:
        if not self.enabled:
            return state

        # --- Post-processing pass: ingest the completed conversation ---
        if state.response is not None:
            user_msgs = [
                m.content for m in state.request.messages if m.role == Role.USER
            ]
            if user_msgs and state.response.text:
                self.ingest(
                    state.request.id,
                    user_msgs[-1],
                    state.response.text,
                )
            return state

        # --- Pre-processing pass: inject relevant context ---
        user_msgs = [
            m.content for m in state.request.messages if m.role == Role.USER
        ]
        if not user_msgs:
            return state

        query = user_msgs[-1]
        try:
            collection = self._ensure_collection()
            if collection.count() == 0:
                return state

            results = collection.query(
                query_texts=[query],
                n_results=min(self.n_results, collection.count()),
            )
            documents = results.get("documents", [[]])[0]
            if not documents:
                return state

            context_block = "\n---\n".join(documents)
            injection = Message(
                role=Role.SYSTEM,
                content=(
                    "The following is relevant context from previous conversations. "
                    "Use it if helpful, but prioritize the current request:\n\n"
                    f"{context_block}"
                ),
            )
            # Insert right after any existing system messages
            insert_idx = 0
            for i, m in enumerate(state.request.messages):
                if m.role == Role.SYSTEM:
                    insert_idx = i + 1
                else:
                    break
            state.request.messages.insert(insert_idx, injection)
            state.extras["rag_injected"] = True
            logger.debug("Injected %d RAG documents", len(documents))

        except Exception:
            logger.exception("RAG injection failed, continuing without context")

        return state
