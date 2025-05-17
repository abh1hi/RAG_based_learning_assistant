import streamlit as st
from test import query_rag, check_vector_store_details, Config
from logger_utils import RAGLogger
import os
import json
from datetime import datetime
import base64  # Add this import

# Initialize logger
logger = RAGLogger()

def display_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")

def main():
    st.set_page_config(
        page_title="Science Tutor RAG System",
        page_icon="🔬",
        layout="wide"
    )

    st.title("📚 Interactive Science Tutor")
    st.subheader("Ask questions about your science textbook!")

    # Sidebar with system info
    with st.sidebar:
        st.header("System Information")
        st.write(f"Model: {Config.MODEL_NAME}")
        st.write(f"PDF: {Config.PDF_PATH}")
        
        if os.path.exists(Config.PERSIST_DIR):
            st.success("Vector Store: Connected")
        else:
            st.error("Vector Store: Not Found")
        
        # Vector store details button
        if st.button("Check Vector Store Details"):
            with st.spinner("Checking vector store..."):
                details = check_vector_store_details()
                with st.expander("Vector Store Details", expanded=True):
                    if "error" in details:
                        st.error(f"Error: {details['error']}")
                    else:
                        st.write("📁 Store Location:", details["Vector store location"])
                        st.write("📚 Document Count:", details["Number of documents"])
                        if "Sample document IDs" in details:
                            st.write("📄 Sample Document IDs:")
                            for doc_id in details["Sample document IDs"]:
                                st.code(doc_id, language="text")
        
        # Query history button
        if st.button("View Query History"):
            try:
                with open(logger.query_history_file, 'r') as f:
                    history = json.load(f)
                st.write("Recent Queries:")
                for query in history[-5:]:  # Show last 5 queries
                    with st.expander(f"Q: {query['question'][:50]}..."):
                        st.write(f"Time: {query['timestamp']}")
                        st.write(f"Answer: {query['answer']}")
            except Exception as e:
                st.error(f"Error loading query history: {str(e)}")
       # Add PDF viewer section
    with st.expander("📑 View Source PDF", expanded=False):
        if os.path.exists(Config.PDF_PATH):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"Currently loaded: {os.path.basename(Config.PDF_PATH)}")
            with col2:
                if st.button("Toggle PDF View"):
                    st.session_state.show_pdf = not st.session_state.get('show_pdf', False)
            
            if st.session_state.get('show_pdf', False):
                display_pdf(Config.PDF_PATH)
        else:
            st.error("PDF file not found!")


    # Main question input area
    question = st.text_input("Enter your science question:", 
                           placeholder="Example: What is rancidity?")

    # Answer generation
    if st.button("Get Answer"):
        if question:
            with st.spinner("Generating answer..."):
                try:
                    answer = query_rag(question)
                    
                    # Create a results container
                    with st.container():
                        st.success("✨ Answer generated successfully!")
                        
                        # Display Q&A in a nice format
                        st.write("### Question")
                        st.info(question)
                        
                        st.write("### Answer")
                        st.markdown(answer)
                        
                        # Add timestamp
                        st.caption(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Log the successful query
                    logger.log_query(
                        question=question,
                        answer=answer,
                        metadata={
                            "model": Config.MODEL_NAME,
                            "pdf": Config.PDF_PATH
                        }
                    )
                except Exception as e:
                    error_msg = f"Error generating answer: {str(e)}"
                    st.error(error_msg)
                    logger.log_error(error_msg, {"question": question})
        else:
            st.warning("Please enter a question!")
            logger.log_error("Empty question submitted")

    # Help section
    with st.expander("How to use this app"):
        st.markdown("""
        1. Enter your science-related question in the text box
        2. Click 'Get Answer' to receive a detailed explanation
        3. The system will provide:
            - A clear, concise answer
            - Supporting examples when relevant
            - Direct quotes from the textbook when available
        """)

if __name__ == "__main__":
    main()
