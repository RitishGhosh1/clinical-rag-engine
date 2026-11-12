from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import re
from app.config import get_settings
from app.services.vector_store import get_vector_store_manager
from app.schemas import Citation, DocumentMetadatafilter

settings = get_settings()

CLINICAL_SYSTEM_PROMPT = """You are an auditable clinical trials intelligence and adverse event verification assistant.
Your outputs are audited by medical officers, regulatory bodies, and biostatisticians.

CRITICAL OPERATIONAL RULES:
1. Rely EXCLUSIVELY on the provided Context excerpts below. Do not assume, extrapolate, or infer beyond the explicit text.
2. If the context contains NO relevant clinical information to address the query, output ONLY:
   "Insufficient clinical documentation provided."
3. If the context answers only PART of a multi-part question, answer the documented portions with exact inline bracketed citations [Doc: filename.pdf, Page: X], and explicitly state which requested aspects are not documented in the provided context.
4. CITATION INTEGRITY:
   - Place inline bracketed citations [Doc: filename.pdf, Page: X] immediately after each clinical claim.
   - Use ONLY document names and page numbers explicitly present in the bracketed headers of the retrieved Context excerpts below. NEVER invent, extrapolate, or cite documents/pages not present in Context.
   - Strictly FORBID generating a trailing "References:" or "Sources:" section at the bottom of the answer text. All citations must be embedded inline directly where claims are made.
5. FORMATTING & PRESENTATION:
   - Format answers using concise bullet points with bold lead-ins (e.g., "- **Safety Profile:** ...").
   - Use Markdown tables (with valid `| --- |` header separators and single-line cells) ONLY when comparing 3 or more structured items (e.g., dosing cohorts, arms, or multi-parameter comparisons). Otherwise, use bullet points.
   - Maintain an objective, concise, and clinical tone. Avoid conversational pleasantries, introductory filler, and concluding remarks.

Context:
{context}

Question:
{question}
"""


def sanitize_source_file(val: Optional[str]) -> Optional[str]:
    """Sanitizes source_file filter input, converting Swagger UI placeholders and blanks to None."""
    if not val or not isinstance(val, str):
        return None
    trimmed = val.strip()
    if trimmed.lower() in ["string", "none", "null", "all", "all indexed documents", ""]:
        return None
    return trimmed


class RagEngine:
    def __init__(self):
        self.vs_manager = get_vector_store_manager()
        self.llm = ChatOpenAI(
            model=settings.GROQ_MODEL_NAME,
            api_key=settings.GROQ_API_KEY,
            base_url=settings.GROQ_BASE_URL,
            temperature=0.0
        )
        self.prompt_template = ChatPromptTemplate.from_template(CLINICAL_SYSTEM_PROMPT)

    def clean_filter_var(self, val: Optional[str]) -> Optional[str]:
        return sanitize_source_file(val)

    def _build_source_filter(
        self,
        filters: Optional[DocumentMetadatafilter]
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, str]]]:
        """Builds ChromaDB where clause and active filter list from source_file filter."""
        if not filters:
            return None, []

        clean_source = sanitize_source_file(filters.source_file)
        if clean_source:
            where_clause = {"source_file": {"$eq": clean_source}}
            active_filters = [{"source_file": clean_source}]
            return where_clause, active_filters

        return None, []

    def _retrieve_chunks(
        self,
        query: str,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        store = self.vs_manager.get_store()

        results_with_scores: List[Tuple[Document, float]] = (
            store.similarity_search_with_relevance_scores(
                query=query,
                k=settings.TOP_K,
                filter=where_filter
            )
        )

        qualifying_docs = [
            doc for doc, score in results_with_scores
            if score >= settings.SIMILARITY_SCORE_THRESOLD
        ]

        return qualifying_docs

    def _prepare_context(
        self,
        docs: List[Document],
    ) -> Tuple[str, List[Citation]]:
        context_parts = []
        citations_map: Dict[str, Citation] = {}

        for doc in docs:
            src = doc.metadata.get("source_file", "Unknown")
            pg = doc.metadata.get("page", 0)

            header = f"[Doc: {src}, Page: {pg}]"
            context_parts.append(f"{header}\n{doc.page_content}\n")

            cite_key = f"{src}_{pg}"
            if cite_key not in citations_map:
                citations_map[cite_key] = Citation(
                    source_file=src,
                    page=pg
                )

        full_context = "\n---\n".join(context_parts)
        return full_context, list(citations_map.values())

    def _filter_cited_sources(
        self,
        answer_text: str,
        candidate_citations: List[Citation]
    ) -> List[Citation]:
        # Match [Doc: filename.pdf, Page: X]
        doc_page_matches = re.findall(
            r"\[Doc:\s*([^,\]]+),\s*Page:\s*(\d+)\]",
            answer_text,
            re.IGNORECASE
        )
        if doc_page_matches:
            cited_pairs = {
                (doc_name.strip().lower(), int(page_str))
                for doc_name, page_str in doc_page_matches
            }
            verified = [
                c for c in candidate_citations
                if (c.source_file.lower(), c.page) in cited_pairs
                or (Path(c.source_file).name.lower(), c.page) in cited_pairs
            ]
            if verified:
                return verified

        # Fallback to page-level matching if LLM omitted doc name or slightly altered format
        cited_pages = set(re.findall(r"Page:\s*(\d+)", answer_text, re.IGNORECASE))
        if not cited_pages:
            return []
        return [
            citation for citation in candidate_citations
            if str(citation.page) in cited_pages
        ]

    def answer_query(
        self,
        question: str,
        explicit_filter: Optional[DocumentMetadatafilter] = None
    ) -> Tuple[str, List[Citation], int, List[Dict[str, str]], str]:
        """Resolves source_file filter, retrieves matching evidence,
        and executes auditable clinical answer synthesis.
        """
        search_query = question.strip()
        where_filter, active_filter_list = self._build_source_filter(explicit_filter)

        # Retrieval using explicit where_filter (source_file) or global semantic search (where_filter=None)
        docs = self._retrieve_chunks(query=search_query, where_filter=where_filter)

        # Strict filter enforcement: NEVER fall back to unfiltered search if 0 chunks matched
        if not docs:
            return (
                "Insufficient clinical documentation provided.",
                [],
                0,
                active_filter_list,
                search_query
            )

        # Context formatting & generation
        context, candidate_citations = self._prepare_context(docs)
        synthesis_chain = self.prompt_template | self.llm
        response = synthesis_chain.invoke({"context": context, "question": search_query})
        answer_text = response.content.strip()

        if answer_text == "Insufficient clinical documentation provided.":
            return answer_text, [], len(docs), active_filter_list, search_query

        # Strip trailing References / Sources section if generated
        answer_text = re.sub(
            r"\n+(?:#+\s*)?(?:References|Sources):\s*$",
            "",
            answer_text,
            flags=re.IGNORECASE
        ).strip()

        verified_citations = self._filter_cited_sources(answer_text, candidate_citations)
        return answer_text, verified_citations, len(docs), active_filter_list, search_query
