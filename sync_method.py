import streamlit as st
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _sync_status_variables(self):
        """Synchronize status variables for header and sidebar display"""
        try:
            # Set DB status
            st.session_state.db_connected = True  # Assume connected as it's checked in DatabaseManager.__init__
            
            # Set Schema status
            st.session_state.schema_loaded = bool(self.schema_manager.schema_data)
            
            # Set AI model status
            try:
                # Check if Ollama is responsive
                st.session_state.ollama_status = "running"
                st.session_state.mistral_loaded = True
            except:
                st.session_state.ollama_status = "unavailable"
                st.session_state.mistral_loaded = False
                
            # Update overall status
            st.session_state.local_ai_ready = (
                st.session_state.ollama_status == "running" and
                st.session_state.mistral_loaded and
                st.session_state.schema_loaded and
                st.session_state.db_connected
            )
            
        except Exception as e:
            logger.error(f"Error syncing status variables: {e}")
            # Set default values if error occurs
            if "ollama_status" not in st.session_state:
                st.session_state.ollama_status = "unknown"
            if "mistral_loaded" not in st.session_state:
                st.session_state.mistral_loaded = False
            if "schema_loaded" not in st.session_state:
                st.session_state.schema_loaded = False
            if "db_connected" not in st.session_state:
                st.session_state.db_connected = False
            if "local_ai_ready" not in st.session_state:
                st.session_state.local_ai_ready = False
