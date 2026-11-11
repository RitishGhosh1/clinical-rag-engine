from typing import Optional, List, Dict, Tuple, Any 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import re
from app.config import get_settings
from app.services.vector_store import get_vector_store_manager
from app.schemas import Citation, DocumentMetadatafilter , ExtractedQueryIntent

settings=get_settings()

QUERY_EXTRACTOR_PROMPT = """You are a clinical query parser for a medical retrieval engine.
Analyze the user's input query to extract specific trial metadata constraints and formulate a clean semantic question.

Rules:
1. Extract any mentioned drug or therapeutic agent into 'drug_name' (normalized to Title Case).
2. Extract trial phase into 'trial_phase' strictly matching these roman numericals or their integer counterparts (I or 1, II or 2, IV or 4 etc): 'Phase I', 'Phase II', 'Phase III', 'Phase IV', or null if unspecified.
3. Formulate 'cleaned_query' by stripping out the explicit phase/drug constraint keywords while keeping the medical/clinical substance (e.g., adverse events, dosage, inclusion criteria).
4. If no specific drug or phase is mentioned, set those fields to null and keep the cleaned_query identical to the user query.

User Query:
{question}
"""

CLINICAL_SYSTEM_PROMPT = """You are an auditable clinical trials intelligence and adverse event verification assistant.
Your outputs are audited by medical officers, regulatory bodies, and biostatisticians.

CRITICAL OPERATIONAL RULES:
1. Rely EXCLUSIVELY on the provided Context excerpts below. Do not assume, extrapolate, or infer beyond the explicit text.
2. If the context contains NO relevant clinical information to address the query, output ONLY:
   "Insufficient clinical documentation provided."
3. If the context answers only PART of a multi-part question, answer the documented portions with exact bracketed citations (e.g., [Doc: sample.pdf, Page: 17]), and explicitly state which requested aspects (e.g., specific calendar timelines or dosages) are not documented in the provided context.
4. CITATION INTEGRITY: You may ONLY cite document names and page numbers that are explicitly stated in the bracketed headers of the provided Context. NEVER cite a page number that does not appear in the context excerpts.
5. Structure the response clearly:
   - If the query covers multiple trial phases, separate findings phase by phase.
   - If filtered to a specific phase, output evidence solely applicable to that phase.
6. Maintain an objective, concise, and clinical tone. Avoid conversational pleasantries.

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

    def clean_filter_var(self,val:Optional[str])->Optional[str]:
        if not val or val.strip().lower() in ["string","none","null",""]:
            return None
        return val.strip()
    
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
            drug=self.clean_filter_var(doc.metadata.get("drug_name"))
            phase=self.clean_filter_var(doc.metadata.get("trial_phase"))

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

    def _filter_cited_sources(
            self,
            answer_text:str,
            candidate_citations: List[Citation]
    ) -> List[Citation]:
        cited_pages=set(re.findall(r"Page:\s*(\d+)",answer_text,re.IGNORECASE))
        if not cited_pages:
            return []
        verified_citations=[
            citation for citation in candidate_citations
            if str(citation.page) in cited_pages
        ]
        return verified_citations

    
    def answer_query(
        self,
        question: str,
        explicit_filter: Optional[DocumentMetadatafilter] = None
    ) -> Tuple[str, List[Citation], int, Optional[DocumentMetadatafilter], str]:
        """Resolves filters (auto or explicit), retrieves matching evidence,

        and executes auditable clinical answer synthesis.
        """
        active_filter=None
        search_query=question
        # Step A: Resolve intent and filter
        if explicit_filter:
            drug=self.clean_filter_var(explicit_filter.drug_name)
            phase=self.clean_filter_var(explicit_filter.trial_phase)
            if drug or phase:
                active_filter=DocumentMetadatafilter(
                    drug_name=drug,
                    trial_phase=phase
                )
                search_query=question

        if active_filter is None:
            extracted = self._extracted_query(question)
            has_extracted_metadata=bool(extracted.drug_name or extracted.trial_phase)
            if has_extracted_metadata:
                active_filter = DocumentMetadatafilter(
                    drug_name=extracted.drug_name,
                    trial_phase=extracted.trial_phase
                )
            search_query = extracted.cleaned_query

        # Step B: Retrieval on the cleaned semantic question with active filter
        docs = self._retrieve_chunks(query=search_query, metadatafilter=active_filter)
        print("\n" + "=" * 30 + " RETRIEVED CHUNKS DEBUG " + "=" * 30)
        for idx, doc in enumerate(docs):
            print(f"\n--- CHUNK {idx+1} [Page {doc.metadata.get('page')}] ---")
            print(doc.page_content[:300] + "...")
        print("=" * 70 + "\n")
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

        verified_citations=self._filter_cited_sources(answer_text,citations)
        return answer_text, verified_citations, len(docs), active_filter, search_query
