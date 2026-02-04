import streamlit as st
def render_progress_bar(self, message: str = None, progress: float = None):
        """Render a progress bar with animation"""
        # Get or use provided message
        thinking_msg = message or st.session_state.thinking_message or "🤖 Processing your request..."
        progress_value = progress if progress is not None else st.session_state.get('progress_value', 0.0)
        
        # Create progress bar container if it doesn't exist
        if not hasattr(self, 'progress_bar') or self.progress_bar is None:
            self.progress_bar = st.progress(0)
        
        # Create status message container if it doesn't exist
        if not hasattr(self, 'status_text') or self.status_text is None:
            self.status_text = st.empty()
        
        # Update progress bar
        self.progress_bar.progress(progress_value)
        
        # Determine phase based on progress
        phase = "⏳ Analyzing your question..."
        if progress_value > 0.33 and progress_value <= 0.66:
            phase = "🧠 Generating SQL..."
        elif progress_value > 0.66:
            phase = "📊 Executing query..."
        
        # Update status text
        percentage = int(progress_value * 100)
        self.status_text.markdown(f'<div class="progress-status">{phase} ({percentage}%)</div>', 
                                unsafe_allow_html=True)
