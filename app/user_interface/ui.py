import streamlit as st
import requests
from typing import Dict, Any

# Configuration matching FastAPI server
API_BASE_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="ClinicalTrialDoc-Insight",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for clinical audit cards (Dark & Light mode adaptive)
st.markdown("""
<style>
    .citation-badge {
        background-color: var(--secondary-background-color, #f0f2f6);
        color: var(--text-color, inherit);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-left: 4px solid #0068c9;
        padding: 8px 14px;
        margin-bottom: 8px;
        border-radius: 4px;
        font-size: 0.9em;
    }

    @media (prefers-color-scheme: dark) {
        .citation-badge {
            background-color: var(--secondary-background-color, #262730);
            color: var(--text-color, #f0f2f6);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-left: 4px solid #388bfd;
        }
    }

    [data-theme="dark"] .citation-badge,
    [data-base-theme="dark"] .citation-badge,
    .stApp[data-theme="dark"] .citation-badge {
        background-color: var(--secondary-background-color, #262730) !important;
        color: var(--text-color, #f0f2f6) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-left: 4px solid #388bfd !important;
    }

    .status-ok { color: #09ab3b; font-weight: 600; }
    .status-err { color: #ff2b2b; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# API Helper Functions
# -----------------------------------------------------------------------------
def fetch_health() -> Dict[str, Any]:
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=5)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        return {"status": "Unreachable", "error": str(e)}


def fetch_documents() -> list:
    try:
        res = requests.get(f"{API_BASE_URL}/documents", timeout=5)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list):
                return data
            return data.get("documents", [])
        return []
    except Exception:
        return []


def delete_document(filename: str) -> bool:
    try:
        res = requests.delete(f"{API_BASE_URL}/documents/{filename}", timeout=10)
        return res.status_code == 200
    except Exception:
        return False


def render_audit_trail(audit: Dict[str, Any], expanded: bool = False):
    """Renders the provenance verification and audit trail expander."""
    with st.expander("🔍 Audit Trail & Provenance Verification", expanded=expanded):
        c1, c2, c3 = st.columns(3)
        c1.metric("Chunks Evaluated", audit.get("chunks_retrieved", 0))

        ext_filter = audit.get("extracted_filter") or []
        scoped_file = None
        if isinstance(ext_filter, list):
            for item in ext_filter:
                if isinstance(item, dict) and "source_file" in item:
                    scoped_file = item["source_file"]
        elif isinstance(ext_filter, dict):
            scoped_file = ext_filter.get("source_file")

        filter_label = scoped_file if scoped_file else "All Documents"
        c2.metric("Document Scope", filter_label)
        c3.metric("Verified Citations", len(audit.get("citations", [])))

        st.caption(f"**Semantic Search Query:** `{audit.get('cleaned_query', '')}`")

        citations = audit.get("citations", [])
        if citations:
            st.markdown("**Referenced Sources:**")
            for cite in citations:
                st.markdown(
                    f"<div class='citation-badge'>"
                    f"<b>Document:</b> {cite['source_file']} &nbsp;|&nbsp; "
                    f"<b>Page:</b> {cite['page']}"
                    f"</div>",
                    unsafe_allow_html=True
                )


# -----------------------------------------------------------------------------
# Sidebar: System Diagnostics & Document Lifecycle Management
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🧬 ClinicalTrialDoc-Insight")
    st.caption("Auditable Clinical Protocol RAG Engine")
    st.divider()

    # 1. Active Health Probe
    st.subheader("System Diagnostics")
    health = fetch_health()
    if health.get("status") == "Healthy":
        st.markdown("<span class='status-ok'>● Online</span>", unsafe_allow_html=True)
        st.caption(f"Indexed Chunks: `{health.get('persisted_documents_count', 0)}`")
        st.caption(f"Embedder: `{health.get('embedding_model', '').split('/')[-1]}`")
        st.caption(f"LLM: `{health.get('llm_model', '')}`")
    else:
        st.markdown("<span class='status-err'>● Offline</span>", unsafe_allow_html=True)
        st.error("FastAPI backend is unreachable. Start Uvicorn on port 8000.")

    st.divider()

    # 2. PDF Ingestion Form
    st.subheader("Upload Protocol PDF")
    with st.form("upload_form", clear_on_submit=True):
        uploaded_file = st.file_uploader("Select Clinical PDF", type=["pdf"])
        submit_upload = st.form_submit_button("Ingest & Index", use_container_width=True)

        if submit_upload:
            if not uploaded_file:
                st.warning("Please select a PDF file first.")
            else:
                with st.spinner("Streaming pages, chunking, and computing CPU embeddings..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    try:
                        resp = requests.post(
                            f"{API_BASE_URL}/documents/upload",
                            files=files,
                            timeout=120
                        )
                        if resp.status_code == 201:
                            res_data = resp.json()
                            st.success(f"Indexed {res_data['total_chunks']} chunks from `{res_data['filename']}`.")
                            st.rerun()
                        else:
                            st.error(f"Upload failed: {resp.json().get('detail', resp.text)}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")

    st.divider()

    # 3. Active Document Repository & Purge Controls
    st.subheader("Indexed Protocols")
    docs = fetch_documents()
    if not docs:
        st.info("No documents currently indexed.")
    else:
        for idx, doc in enumerate(docs):
            if isinstance(doc, dict):
                fname = doc.get("filename", "Unknown")
                chunks = doc.get("chunks_count", 0)
            else:
                fname = str(doc)
                chunks = "?"

            with st.container(border=True):
                st.markdown(f"**📄 {fname}**")
                st.caption(f"Chunks: `{chunks}`")

                if st.button("🗑️ Delete from Index", key=f"del_{fname}_{idx}", use_container_width=True):
                    with st.spinner(f"Purging {fname}..."):
                        if delete_document(fname):
                            st.toast(f"Deleted {fname}")
                            st.rerun()
                        else:
                            st.error("Failed to delete document.")


# -----------------------------------------------------------------------------
# Main Workspace: Clinical Query & Audit Interface
# -----------------------------------------------------------------------------
st.header("Clinical Evidence & Adverse Event Verification")
st.markdown(
    "Ask natural language questions across indexed trial protocols and clinical documentation. "
    "Every claim is strictly bound to verified page citations."
)

# Document Scoping Dropdown Selector
docs_list = fetch_documents()
doc_filenames = []
for d in docs_list:
    if isinstance(d, dict) and d.get("filename"):
        doc_filenames.append(d["filename"])
    elif isinstance(d, str):
        doc_filenames.append(d)

scope_options = ["All Indexed Documents"] + sorted(list(set(doc_filenames)))
selected_scope = st.selectbox(
    "🎯 Document Scope",
    options=scope_options,
    index=0,
    help="Scope retrieval to a single indexed document or search globally across all indexed protocols."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "audit" in msg:
            render_audit_trail(msg["audit"], expanded=False)

if prompt := st.chat_input("e.g., Why was the surrogate endpoint misleading for survival?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving matching clinical chunks and verifying citations..."):
            query_payload = {"question": prompt}

            if selected_scope != "All Indexed Documents":
                query_payload["filter_metadata"] = {"source_file": selected_scope}

            try:
                response = requests.post(
                    f"{API_BASE_URL}/query",
                    json=query_payload,
                    timeout=60
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    st.markdown(answer)

                    audit_payload = {
                        "citations": data.get("citations", []),
                        "chunks_retrieved": data.get("chunks_retrieved", 0),
                        "extracted_filter": data.get("extracted_filter", []),
                        "cleaned_query": data.get("cleaned_query", prompt)
                    }

                    render_audit_trail(audit_payload, expanded=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "audit": audit_payload
                    })
                else:
                    err_msg = f"Backend Error ({response.status_code}): {response.json().get('detail', response.text)}"
                    st.error(err_msg)
            except Exception as e:
                st.error(f"Failed to reach backend server: {str(e)}")
