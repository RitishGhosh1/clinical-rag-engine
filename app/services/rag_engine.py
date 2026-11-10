from typing import Optional, List, Dict, Tuple, Any 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from app.config import get_settings
from app.services.vector_store import get_vector_store_manager
from app.schemas import Citation, DocumentMetadatafilter , ExtractedQueryIntent

settings=get_settings()

QUERY_EXTRACTOR_PROMPT = """You are a clinical query parser for a medical retrieval engine.
Analyze the user's input query to extract specific trial metadata constraints and formulate a clean semantic question.

Rules:
1. Extract any mentioned drug or therapeutic agent into 'drug_name' (normalized to Title Case).
2. Extract trial phase into 'trial_phase' strictly matching: 'Phase I', 'Phase II', 'Phase III', 'Phase IV', or null if unspecified.
3. Formulate 'cleaned_query' by stripping out the explicit phase/drug constraint keywords while keeping the medical/clinical substance (e.g., adverse events, dosage, inclusion criteria).
4. If no specific drug or phase is mentioned, set those fields to null and keep the cleaned_query identical to the user query.

User Query:
{question}
"""

CLINICAL_SYSTEM_PROMPT = """You are an auditable clinical trials intelligence and adverse event verification assistant.
Your outputs are audited by medical officers, regulatory bodies, and biostatisticians.

CRITICAL OPERATIONAL RULES:
1. Rely EXCLUSIVELY on the provided Context excerpts below. Do not assume, extrapolate, or infer beyond the explicit text.
2. If the context does not explicitly provide the exact evidence required to answer the question, output ONLY the exact phrase:
   "Insufficient clinical documentation provided."
3. Every clinical fact, adverse event, incidence rate, dosage, or exclusion criteria stated MUST include inline bracketed page citations matching the provided excerpts (e.g., [Doc: study_protocol.pdf, Page: 12]).
4. Structure the response clearly:
   - If the query covers multiple trial phases, separate the findings phase by phase (e.g., Phase I, Phase II, Phase III).
   - If filtered to a specific phase, output evidence solely applicable to that phase.
5. Maintain an objective, concise, and clinical tone. Avoid conversational pleasantries.

Context:
{context}

Question:
{question}
"""

class RagEngine:
    def __init__(self):
        self.vs_manager=get_vector_store_manager()
        self.llm=ChatOpenAI(
            model=settings.GROQ_MODEL_NAME,
            api_key=settings.GROQ_API_KEY,
            base_url=settings.GROQ_BASE_URL,
            temperature=0.0
        )
        #FOR INGESTING THE QUERY INTO THE METADATA FILTERS
        self.extractor_prompt=ChatPromptTemplate.from_template(QUERY_EXTRACTOR_PROMPT)
        self.structured_prompt=self.llm.with_structured_output(ExtractedQueryIntent)
        self.extractor_chain=self.extractor_prompt | self.structured_prompt
        self.prompt_template=ChatPromptTemplate.from_template(CLINICAL_SYSTEM_PROMPT)


    def _extracted_query(
            self,
            raw_query:str
    ) -> ExtractedQueryIntent:
        try:
            extracted: ExtractedQueryIntent = self.extractor_chain.invoke({"question":raw_query})
            return extracted
        except Exception:
            return ExtractedQueryIntent(cleaned_query=raw_query, drug_name=None, trial_phase=None)
        
    def _buld_metadata_filter(
            self,
            filters: Optional[DocumentMetadatafilter]
    ) -> Optional[Dict[str, Any]]:
        if not filters:
            return None
        conditions=[]
        if filters.drug_name:
            conditions.append({"drug_name":{"$eq":filters.drug_name.strip()}})
        if filters.trial_phase:
            conditions.append({"trial_phase":{"$eq":filters.trial_phase.strip()}})

        if not conditions:
            return None
        return conditions[0] if len(conditions)==1 else {"$and":conditions}

    def _retrieve_chunks(
            self,
            query:str,
            metadatafilter: Optional[DocumentMetadatafilter]
    )->List[Document]:
        
        where_filter=self._buld_metadata_filter(metadatafilter) if metadatafilter else None
        store=self.vs_manager.get_store()

        # similarity_search_with_relevance_scores returns (Document, cosine_similarity_score)
        results_with_scores: List[Tuple[Document, float]]=(
            store.similarity_search_with_relevance_scores(
                query=query,
                k=settings.TOP_K,
                filter=where_filter
            )
        )

        qualifying_docs=[
            doc for doc,score in results_with_scores if score>=settings.SIMILARITY_SCORE_THRESOLD
        ]

        return qualifying_docs

    def _prepare_context(
            self,
            docs:List[Document],
    ) ->  Tuple[str, List[Citation]]:
        context_parts = []
        citations_map : Dict[str, Citation]= {}

        for doc in docs:
            src = doc.metadata.get("source_file","Unknown")
            pg = doc.metadata.get("page",0)
            drug=doc.metadata.get("drug_name")
            phase=doc.metadata.get("trial_phase")

            context_parts.append(
                f"[Doc: {src}, Page: {pg}, Drug: {drug or "N/A"}, Phase: {phase or "N/A"}]\n"
                f"{doc.page_content}\n"
            )

            cite_key=f"{src}_{pg}"

            if cite_key not in citations_map:
                citations_map[cite_key]=Citation(
                    source_file=src,
                    page=pg,
                    drug_name=drug,
                    trial_phase=phase
                )

        full_context="\n---\n".join(context_parts)
        return full_context, list(citations_map.values())

    def answer_query(
        self,
        question: str,
        explicit_filter: Optional[DocumentMetadatafilter] = None
    ) -> Tuple[str, List[Citation], int, Optional[DocumentMetadatafilter], str]:
        """Resolves filters (auto or explicit), retrieves matching evidence,

        and executes auditable clinical answer synthesis.
        """
        # Step A: Resolve intent and filter
        if explicit_filter and (explicit_filter.drug_name or explicit_filter.trial_phase):
            active_filter = explicit_filter
            search_query = question
        else:
            extracted = self._extracted_query(question)
            active_filter = DocumentMetadatafilter(
                drug_name=extracted.drug_name,
                trial_phase=extracted.trial_phase
            ) if (extracted.drug_name or extracted.trial_phase) else None
            search_query = extracted.cleaned_query

        # Step B: Retrieval on the cleaned semantic question with active filter
        docs = self._retrieve_chunks(query=search_query, metadatafilter=active_filter)

        if not docs:
            return (
                "Insufficient clinical documentation provided.",
                [],
                0,
                active_filter,
                search_query
            )

        # Step C: Context Formatting & Generation
        context, citations = self._prepare_context(docs)
        self.synthesis_chain=self.prompt_template | self.llm
        response = self.synthesis_chain.invoke({"context": context, "question": question})
        answer_text = response.content.strip()

        if answer_text == "Insufficient clinical documentation provided.":
            return answer_text, [], len(docs), active_filter, search_query

        return answer_text, citations, len(docs), active_filter, search_query
