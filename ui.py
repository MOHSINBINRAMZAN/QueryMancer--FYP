"""
Enhanced UI components for QueryMancer - Local AI SQL Chatbot
This module provides comprehensive UI elements for the Streamlit interface
with Ollama + Mistral + RAG integration and local schema-based SQL generation.

Current Date and Time: 2025-08-19 13:10:26 UTC
Current User: Mohsin Ramzan
"""

import streamlit as st
import pandas as pd
import re
import time
import json
import logging
import os
import asyncio
import threading
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
from pathlib import Path
import hashlib
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


# Ensure logs directory exists
import os
os.makedirs("logs", exist_ok=True)

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/querymancer_ui.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("querymancer.ui")

# Current time and user info
CURRENT_USER = "Mohsin Ramzan"
CURRENT_DATETIME = datetime.now()
CURRENT_DAY = CURRENT_DATETIME.day

class AccuracyLevel(Enum):
    """Accuracy level enumeration"""
    EXCELLENT = "excellent"  # 95%+
    VERY_GOOD = "very_good"  # 90-94%
    GOOD = "good"           # 80-89%
    FAIR = "fair"           # 70-79%
    POOR = "poor"           # Below 70%

@dataclass
class QueryAnalytics:
    """Query analytics data structure"""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    average_execution_time: float = 0.0
    average_confidence: float = 0.0
    accuracy_score: float = 0.0
    sql_generation_accuracy: float = 0.0
    execution_success_rate: float = 0.0
    timestamp: datetime = field(default_factory=lambda: CURRENT_DATETIME)

# Enhanced UI Theme for local AI chatbot
UI_THEME = {
    'primary': '#00d4ff',
    'secondary': '#ff6b6b', 
    'accent': '#4ecdc4',
    'success': '#50fa7b',
    'warning': '#f1fa8c',
    'error': '#ff5555',
    'background': '#0f0f23',
    'surface': '#1a1a2e',
    'text': '#ffffff',
    'text_secondary': '#94a3b8',
    'accuracy_excellent': '#22c55e',
    'accuracy_good': '#3b82f6',
    'accuracy_fair': '#f59e0b',
    'accuracy_poor': '#ef4444',
    'ollama_color': '#8b5cf6',
    'mistral_color': '#ff7849'
}

LOADING_MESSAGES = [
    "🤖 Mistral is analyzing your question...",
    "📋 Loading schema context...",
    "⚙️ Generating SQL with LangChain...",
    "🚀 Executing query on AWS SQL Server...",
    "📊 Processing results...",
    "🎯 Working with local Ollama...",
    "💫 AI is thinking locally...",
    "🔮 Mistral is crafting your query...",
    "🧠 Local inference in progress...",
    "⚡ Secure local processing...",
    "🎪 Local AI magic happening...",
    "📊 Analyzing database schema..."
]

AI_AVATARS = ["🤖", "🧠", "⚡", "🔮", "🎯", "🚀", "💡", "🔬", "📊", "🎪", "🌟", "⭐"]
USER_AVATARS = ["👤", "🧑", "👨", "👩", "🙋", "🙋‍♂️", "🙋‍♀️", "🤵", "👨‍💼", "👩‍💼", "🎓", "💼"]

# Sample query suggestions for different business scenarios

class EnhancedLocalUI:
    """Enhanced UI class for local AI SQL Chatbot"""
    
    def __init__(self):
        """Initialize enhanced UI components for local deployment"""
        self.theme = UI_THEME
        self.user = CURRENT_USER
        self.current_time = CURRENT_DATETIME
        self.setup_complete = False
        self.analytics = QueryAnalytics()
        self.accuracy_target = 0.95  # 95% accuracy target for local AI
        self.performance_threshold = 5.0  # 5 seconds max for local processing
        
    def setup(self):
        """Set up the enhanced UI components for local AI chatbot"""
        if self.setup_complete:
            return
        
        # Set enhanced page config
        st.set_page_config(
            page_title="QueryMancer 🤖 - Local AI SQL Chatbot (Ollama + Mistral + RAG with FAISS vector DB)",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': None,
                'Report a bug': None,
                'About': 'QueryMancer - Local AI SQL Chatbot with Ollama + Mistral + RAG with FAISS vector DB   '
            }
        )
        
        # Initialize enhanced session state
        self._init_enhanced_session_state()
        
        # Apply enhanced custom CSS
        self._apply_enhanced_css()
        
        # Initialize local AI status
        self._init_local_ai_status()
        
        self.setup_complete = True
    
    def _init_enhanced_session_state(self):
        """Initialize enhanced Streamlit session state for local AI"""
        default_session_values = {
            # Chat history with enhanced metadata
            "messages": [],
            "user_avatar": USER_AVATARS[0],
            "ai_avatar": AI_AVATARS[0],
            
            # Query history with local AI analytics
            "query_history": [],
            "last_query": "",
            "last_sql": "",
            "last_result": None,
            "last_confidence": 0.0,
            "last_accuracy": 0.0,
            
            # Local AI specific state
            "ollama_status": "unknown",
            "mistral_loaded": False,
            "schema_loaded": False,
            "db_connected": False,
            "local_ai_ready": False,
            
            # Enhanced UI state
            "show_sql_editor": False,
            "show_schema_explorer": False,
            "show_query_builder": False,
            "show_analytics_dashboard": False,
            "show_local_ai_status": True,
            "current_table": None,
            "selected_database": None,
            "show_welcome": True,
            "show_sample_queries": False,
            
            # Enhanced settings for local AI
            "show_timestamps": True,
            "enable_animations": True,
            "dark_mode": True,
            "show_schema_info": True,
            "show_query_analysis": True,
            "auto_optimize": True,
            "confidence_threshold": 0.8,
            "max_query_time": 10.0,
            "show_sql_in_response": True,
            "enable_query_validation": True,
            "show_execution_stats": True,
            
            # Performance tracking for local AI
            "execution_times": [],
            "confidence_scores": [],
            "accuracy_scores": [],
            "success_count": 0,
            "error_count": 0,
            "sql_generation_successes": 0,
            
            # Local AI metrics
            "ollama_response_times": [],
            "schema_load_time": 0.0,
            "total_tokens_used": 0,
            "average_tokens_per_query": 0,
            
            # Enhanced metadata
            "start_time": CURRENT_DATETIME,
            "current_user": CURRENT_USER,
            "session_id": hashlib.md5(f"{CURRENT_USER}_{CURRENT_DATETIME}".encode()).hexdigest()[:8],
            "total_session_time": 0.0,
            "queries_per_minute": 0.0,
            "local_processing": True,
            
            # Thinking state
            "is_thinking": False,
            "thinking_message": "",
            "processing_start_time": None,
            
            # Authentication state
            "authenticated": False,
            "logged_in_user": "",
            "login_error": ""
        }
        
        # Set default values if not already in session state
        for key, value in default_session_values.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def _apply_enhanced_css(self):
        """Apply enhanced CSS styling for local AI chatbot"""
        # Import the CSS loading utilities
        from load_css import load_custom_css, apply_theme_variables
        
        # Try to load CSS from the external style.css file
        css_loaded = load_custom_css(self.theme)
        
        # If CSS couldn't be loaded, fall back to just applying theme variables
        if not css_loaded:
            apply_theme_variables(self.theme)
            logger.warning("Using fallback CSS styling since style.css couldn't be loaded")
    
    def _init_local_ai_status(self):
        """Initialize local AI status monitoring"""
        if "local_ai_initialized" not in st.session_state:
            st.session_state.local_ai_initialized = True
            self._check_local_ai_components()
    
    def _check_local_ai_components(self):
        """Check status of local AI components"""
        try:
            # First make sure all required session state variables are initialized
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
                
            # Try to import config modules, but handle gracefully if they don't exist
            try:
                # Import your config and tools modules
                from config import check_ollama_status, check_mistral_model, test_db_connection
                from tools import load_schema
                
                # Check Ollama status
                st.session_state.ollama_status = check_ollama_status()
                
                # Check if Mistral model is loaded
                st.session_state.mistral_loaded = check_mistral_model()
            except ImportError as e:
                logger.error(f"Error checking local AI components: {e}")
                # Keep default values for status
            
            # Check schema loading
            schema = load_schema()
            st.session_state.schema_loaded = bool(schema)
            
            # Check database connection
            st.session_state.db_connected = test_db_connection()
            
            # Overall AI readiness
            st.session_state.local_ai_ready = (
                st.session_state.ollama_status == "running" and
                st.session_state.mistral_loaded and
                st.session_state.schema_loaded and
                st.session_state.db_connected
            )
            
        except Exception as e:
            logger.error(f"Error checking local AI components: {e}")
            st.session_state.local_ai_ready = False
    
    def render_enhanced_header(self):
        """Render the enhanced application header for local AI"""
        # Ensure session state variables exist
        if "local_ai_ready" not in st.session_state:
            st.session_state.local_ai_ready = False
        if "ollama_status" not in st.session_state:
            st.session_state.ollama_status = "unknown" 
        if "mistral_loaded" not in st.session_state:
            st.session_state.mistral_loaded = False
        if "schema_loaded" not in st.session_state:
            st.session_state.schema_loaded = False
        if "db_connected" not in st.session_state:
            st.session_state.db_connected = False
            
        # Check local AI status
        ai_status = "🟢 Ready" if st.session_state.local_ai_ready else "🔴 Not Ready"

        ollama_status_color = "🟢" if st.session_state.ollama_status == "running" else "🔴"
        mistral_status_color = "🟢" if st.session_state.mistral_loaded else "🔴"
        schema_status_color = "🟢" if st.session_state.schema_loaded else "🔴"
        db_status_color = "🟢" if st.session_state.db_connected else "🔴"
        
        # Enhanced header with beautiful animations
        st.markdown(f"""
        <style>
            .enhanced-header {{
                background: linear-gradient(135deg, rgba(2, 2, 8, 0.99) 0%, rgba(5, 5, 15, 0.97) 100%);
                backdrop-filter: blur(25px);
                border: 1px solid rgba(0, 212, 255, 0.15);
                border-radius: 24px;
                padding: 2rem 2.5rem;
                margin-bottom: 2rem;
                position: relative;
                overflow: hidden;
                animation: headerSlideIn 0.8s cubic-bezier(0.4, 0, 0.2, 1);
            }}
            
            .enhanced-header::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #00d4ff, #8b5cf6, #ff2d92, #ff6b35, #feca57, #00d4ff);
                background-size: 400% 100%;
                animation: headerGradient 8s ease infinite;
            }}
            
            .enhanced-header::after {{
                content: '';
                position: absolute;
                inset: 0;
                background: linear-gradient(45deg, transparent 40%, rgba(255,255,255,0.015) 50%, transparent 60%);
                animation: headerShimmer 4s ease-in-out infinite;
                pointer-events: none;
            }}
            
            @keyframes headerSlideIn {{
                from {{ opacity: 0; transform: translateY(-30px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            
            @keyframes headerGradient {{
                0% {{ background-position: 0% 50%; }}
                50% {{ background-position: 100% 50%; }}
                100% {{ background-position: 0% 50%; }}
            }}
            
            @keyframes headerShimmer {{
                0% {{ transform: translateX(-100%); }}
                100% {{ transform: translateX(100%); }}
            }}
            
            .header-main {{
                text-align: center;
                margin-bottom: 1.5rem;
                position: relative;
                z-index: 1;
            }}
            
            .header-logo {{
                font-size: 3.5rem;
                margin-bottom: 0.5rem;
                display: inline-block;
                animation: logoFloat 3s ease-in-out infinite;
                filter: drop-shadow(0 0 25px rgba(0, 212, 255, 0.5));
            }}
            
            @keyframes logoFloat {{
                0%, 100% {{ transform: translateY(0) rotate(0deg); }}
                50% {{ transform: translateY(-8px) rotate(5deg); }}
            }}
            
            .header-brand {{
                font-family: 'Orbitron', sans-serif;
                font-size: 2.5rem;
                font-weight: 800;
                background: linear-gradient(135deg, #00d4ff 0%, #8b5cf6 35%, #ff2d92 70%, #ff6b35 100%);
                background-size: 300% 100%;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                letter-spacing: 3px;
                animation: brandGradient 6s ease infinite;
                margin: 0;
            }}
            
            @keyframes brandGradient {{
                0% {{ background-position: 0% 50%; }}
                50% {{ background-position: 100% 50%; }}
                100% {{ background-position: 0% 50%; }}
            }}
            
            .header-tagline {{
                color: #7a7a8a;
                font-size: 1rem;
                letter-spacing: 2px;
                text-transform: uppercase;
                margin-top: 0.5rem;
                font-weight: 400;
            }}
            
            .header-tagline span {{
                color: #00d4ff;
            }}
            
            .status-bar {{
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 0.8rem;
                position: relative;
                z-index: 1;
            }}
            
            .status-badge {{
                background: linear-gradient(135deg, rgba(57, 255, 20, 0.15) 0%, rgba(0, 200, 100, 0.1) 100%);
                border: 1px solid rgba(57, 255, 20, 0.4);
                padding: 0.5rem 1.2rem;
                border-radius: 50px;
                font-size: 0.85rem;
                font-weight: 600;
                color: #39ff14;
                animation: badgeGlow 2s ease-in-out infinite;
            }}
            
            @keyframes badgeGlow {{
                0%, 100% {{ box-shadow: 0 0 10px rgba(57, 255, 20, 0.3); }}
                50% {{ box-shadow: 0 0 25px rgba(57, 255, 20, 0.5); }}
            }}
            
            .status-pill {{
                background: rgba(0, 0, 0, 0.25);
                padding: 0.45rem 1rem;
                border-radius: 25px;
                font-size: 0.85rem;
                color: #9090a0;
                border: 1px solid rgba(255, 255, 255, 0.08);
                transition: all 0.3s ease;
            }}
            
            .status-pill:hover {{
                background: rgba(0, 212, 255, 0.1);
                border-color: rgba(0, 212, 255, 0.3);
                color: #c0c0d0;
                transform: translateY(-2px);
            }}
        </style>
        
        <div class="enhanced-header">
            <div class="header-main">
                <div class="header-logo">🤖</div>
                <h1 class="header-brand">QueryMancer</h1>
                <p class="header-tagline">Local AI SQL Chatbot • <span>Ollama</span> + <span>Mistral</span> + <span>LangChain</span></p>
            </div>
            <div class="status-bar">
                <div class="status-badge">🔐 100% Local Processing</div>
                <div class="status-pill">{ollama_status_color} Ollama: {st.session_state.ollama_status}</div>
                <div class="status-pill">{mistral_status_color} Mistral: {'Loaded' if st.session_state.mistral_loaded else 'Not Loaded'}</div>
                <div class="status-pill">{schema_status_color} Schema: {'Ready' if st.session_state.schema_loaded else 'Loading...'}</div>
                <div class="status-pill">{db_status_color} Database: {'Connected' if st.session_state.db_connected else 'Disconnected'}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show refresh button if not all components are ready
        if not st.session_state.local_ai_ready:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🔄 Refresh Status", key="refresh_status"):
                    self._check_local_ai_components()
                    st.rerun()


    def render_login_form(self) -> bool:
        """Render the login form and return True if authenticated"""
        # Check if already authenticated
        if st.session_state.get('authenticated', False):
            return True
        
        # Apply enhanced login-specific CSS with animations
        st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
            
            .main > div { padding-top: 0 !important; }
            header { visibility: hidden; }
            #MainMenu { visibility: hidden; }
            footer { visibility: hidden; }
            .block-container { 
                padding-top: 0 !important; 
                max-width: 100% !important;
                background: linear-gradient(135deg, #000000 0%, #010103 25%, #020206 50%, #010102 75%, #000000 100%);
                min-height: 100vh;
            }
            
            /* Animated Background */
            .login-bg {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
                z-index: -1;
                background: #000000;
            }
            
            .login-bg::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: 
                    radial-gradient(ellipse at 20% 80%, rgba(0, 212, 255, 0.12) 0%, transparent 50%),
                    radial-gradient(ellipse at 80% 20%, rgba(255, 45, 146, 0.1) 0%, transparent 50%),
                    radial-gradient(ellipse at 40% 40%, rgba(139, 92, 246, 0.08) 0%, transparent 50%),
                    radial-gradient(ellipse at 60% 70%, rgba(78, 205, 196, 0.06) 0%, transparent 50%);
                animation: bgFloat 20s ease-in-out infinite;
            }
            
            @keyframes bgFloat {
                0%, 100% { transform: translate(0, 0) rotate(0deg); }
                25% { transform: translate(-3%, 3%) rotate(90deg); }
                50% { transform: translate(3%, -3%) rotate(180deg); }
                75% { transform: translate(-2%, -2%) rotate(270deg); }
            }
            
            /* Floating Orbs */
            .orb {
                position: fixed;
                border-radius: 50%;
                filter: blur(50px);
                opacity: 0.25;
                animation: orbFloat 15s ease-in-out infinite;
            }
            
            .orb-1 {
                width: 300px;
                height: 300px;
                background: linear-gradient(135deg, #00d4ff, #0099cc);
                top: 10%;
                left: 10%;
                animation-delay: 0s;
            }
            
            .orb-2 {
                width: 250px;
                height: 250px;
                background: linear-gradient(135deg, #ff2d92, #cc0066);
                bottom: 15%;
                right: 10%;
                animation-delay: -5s;
            }
            
            .orb-3 {
                width: 200px;
                height: 200px;
                background: linear-gradient(135deg, #8b5cf6, #6b21a8);
                top: 50%;
                right: 20%;
                animation-delay: -10s;
            }
            
            @keyframes orbFloat {
                0%, 100% { transform: translate(0, 0) scale(1); }
                25% { transform: translate(50px, -50px) scale(1.1); }
                50% { transform: translate(-30px, 30px) scale(0.9); }
                75% { transform: translate(40px, 40px) scale(1.05); }
            }
            
            /* Login Card Styles */
            .login-card {
                background: linear-gradient(135deg, rgba(2, 2, 8, 0.98) 0%, rgba(5, 5, 15, 0.96) 100%);
                backdrop-filter: blur(30px);
                border-radius: 28px;
                padding: 3.5rem;
                margin-top: 3rem;
                box-shadow: 
                    0 30px 100px rgba(0, 0, 0, 0.8),
                    0 0 60px rgba(0, 212, 255, 0.1),
                    inset 0 1px 0 rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(0, 212, 255, 0.15);
                position: relative;
                overflow: hidden;
                animation: cardAppear 0.8s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            @keyframes cardAppear {
                from {
                    opacity: 0;
                    transform: translateY(60px) scale(0.9);
                }
                to {
                    opacity: 1;
                    transform: translateY(0) scale(1);
                }
            }
            
            .login-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #00d4ff, #8b5cf6, #ff2d92, #ff6b35, #00d4ff);
                background-size: 400% 100%;
                animation: gradientFlow 6s ease infinite;
            }
            
            .login-card::after {
                content: '';
                position: absolute;
                inset: 0;
                background: linear-gradient(45deg, transparent 40%, rgba(255,255,255,0.015) 50%, transparent 60%);
                animation: cardShimmer 5s ease-in-out infinite;
                pointer-events: none;
            }
            
            @keyframes cardShimmer {
                0% { transform: translateX(-100%) rotate(45deg); }
                100% { transform: translateX(200%) rotate(45deg); }
            }
            
            @keyframes gradientFlow {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            
            /* Logo Animation */
            .login-logo {
                font-size: 5.5rem;
                text-align: center;
                margin-bottom: 1.5rem;
                animation: logoFloat 4s ease-in-out infinite, logoGlow 3s ease-in-out infinite alternate;
                filter: drop-shadow(0 0 35px rgba(0, 212, 255, 0.6));
            }
            
            @keyframes logoFloat {
                0%, 100% { transform: translateY(0) rotate(0deg); }
                25% { transform: translateY(-12px) rotate(3deg); }
                75% { transform: translateY(-6px) rotate(-3deg); }
            }
            
            @keyframes logoGlow {
                0% { filter: drop-shadow(0 0 35px rgba(0, 212, 255, 0.7)); }
                100% { filter: drop-shadow(0 0 55px rgba(255, 45, 146, 0.9)); }
            }
            
            /* Title Animation */
            .login-title {
                font-family: 'Orbitron', sans-serif !important;
                font-size: 2.8rem !important;
                font-weight: 800 !important;
                text-align: center;
                background: linear-gradient(135deg, #00d4ff 0%, #8b5cf6 25%, #ff2d92 50%, #ff6b35 75%, #feca57 100%);
                background-size: 400% 100%;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 0.5rem;
                letter-spacing: 4px;
                animation: titleGradient 8s ease infinite, titlePulse 4s ease-in-out infinite;
            }
            
            @keyframes titleGradient {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            
            @keyframes titlePulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.02); }
            }
            
            /* Subtitle */
            .login-subtitle {
                color: #8a8a9a;
                text-align: center;
                font-size: 1.05rem;
                margin-bottom: 2.5rem;
                font-weight: 400;
                letter-spacing: 1px;
            }
            
            .login-subtitle span {
                color: #00d4ff;
                font-weight: 500;
            }
            
            /* Features Pills */
            .features-pills {
                display: flex;
                justify-content: center;
                gap: 0.8rem;
                flex-wrap: wrap;
                margin-bottom: 2rem;
            }
            
            .pill {
                background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
                border: 1px solid rgba(0, 212, 255, 0.3);
                padding: 0.5rem 1rem;
                border-radius: 50px;
                font-size: 0.85rem;
                color: #a0a0b0;
                transition: all 0.3s ease;
            }
            
            .pill:hover {
                border-color: #00d4ff;
                color: #00d4ff;
                transform: translateY(-2px);
                box-shadow: 0 5px 20px rgba(0, 212, 255, 0.2);
            }
            
            /* Footer */
            .login-footer {
                text-align: center;
                margin-top: 2.5rem;
                padding-top: 1.5rem;
                border-top: 1px solid rgba(255, 255, 255, 0.05);
            }
            
            .login-footer p {
                color: #6a6a7a;
                font-size: 0.9rem;
                margin: 0.3rem 0;
            }
            
            .login-footer .highlight {
                color: #00d4ff;
                font-weight: 500;
            }
            
            .demo-hint {
                background: linear-gradient(135deg, rgba(57, 255, 20, 0.1) 0%, rgba(0, 212, 255, 0.1) 100%);
                border: 1px solid rgba(57, 255, 20, 0.3);
                padding: 0.6rem 1.2rem;
                border-radius: 10px;
                margin-top: 1rem;
                display: inline-block;
            }
            
            .demo-hint code {
                background: rgba(0, 0, 0, 0.3);
                padding: 0.2rem 0.5rem;
                border-radius: 4px;
                color: #39ff14;
                font-family: 'JetBrains Mono', monospace;
            }
            
            /* Enhanced Floating Particles */
            .floating-particles {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 1;
                overflow: hidden;
            }
            
            .particle {
                position: absolute;
                width: 6px;
                height: 6px;
                border-radius: 50%;
                animation: floatUp 15s infinite linear;
            }
            
            .particle:nth-child(1) { background: #00d4ff; left: 5%; animation-delay: 0s; animation-duration: 12s; }
            .particle:nth-child(2) { background: #8b5cf6; left: 10%; animation-delay: -2s; animation-duration: 14s; }
            .particle:nth-child(3) { background: #ff2d92; left: 20%; animation-delay: -4s; animation-duration: 16s; }
            .particle:nth-child(4) { background: #39ff14; left: 30%; animation-delay: -6s; animation-duration: 13s; }
            .particle:nth-child(5) { background: #feca57; left: 40%; animation-delay: -8s; animation-duration: 15s; }
            .particle:nth-child(6) { background: #00f5ff; left: 50%; animation-delay: -10s; animation-duration: 17s; }
            .particle:nth-child(7) { background: #ff6b6b; left: 60%; animation-delay: -12s; animation-duration: 14s; }
            .particle:nth-child(8) { background: #4ecdc4; left: 70%; animation-delay: -14s; animation-duration: 16s; }
            .particle:nth-child(9) { background: #a855f7; left: 80%; animation-delay: -3s; animation-duration: 13s; }
            .particle:nth-child(10) { background: #14b8a6; left: 90%; animation-delay: -7s; animation-duration: 15s; }
            .particle:nth-child(11) { background: #f43f5e; left: 15%; animation-delay: -9s; animation-duration: 18s; }
            .particle:nth-child(12) { background: #3b82f6; left: 85%; animation-delay: -11s; animation-duration: 14s; }
            
            @keyframes floatUp {
                0% {
                    transform: translateY(100vh) scale(0) rotate(0deg);
                    opacity: 0;
                }
                10% { opacity: 1; }
                90% { opacity: 1; }
                100% {
                    transform: translateY(-100vh) scale(1.5) rotate(720deg);
                    opacity: 0;
                }
            }
            
            /* Cyber Scanline Effect */
            .cyber-scanline {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 2;
            }
            
            .cyber-scanline::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.8), transparent);
                animation: scanDown 4s linear infinite;
            }
            
            @keyframes scanDown {
                0% { top: -4px; opacity: 0; }
                10% { opacity: 1; }
                90% { opacity: 1; }
                100% { top: 100%; opacity: 0; }
            }
            
            /* Pulsing Ring Effect */
            .pulse-rings {
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                pointer-events: none;
                z-index: 0;
            }
            
            .pulse-ring {
                position: absolute;
                border-radius: 50%;
                border: 2px solid rgba(0, 212, 255, 0.3);
                animation: pulseExpand 4s ease-out infinite;
            }
            
            .pulse-ring:nth-child(1) { width: 100px; height: 100px; margin: -50px 0 0 -50px; animation-delay: 0s; }
            .pulse-ring:nth-child(2) { width: 100px; height: 100px; margin: -50px 0 0 -50px; animation-delay: 1s; }
            .pulse-ring:nth-child(3) { width: 100px; height: 100px; margin: -50px 0 0 -50px; animation-delay: 2s; }
            .pulse-ring:nth-child(4) { width: 100px; height: 100px; margin: -50px 0 0 -50px; animation-delay: 3s; }
            
            @keyframes pulseExpand {
                0% { 
                    transform: scale(1);
                    opacity: 0.8;
                    border-color: rgba(0, 212, 255, 0.5);
                }
                50% { border-color: rgba(139, 92, 246, 0.3); }
                100% { 
                    transform: scale(8);
                    opacity: 0;
                    border-color: rgba(255, 45, 146, 0.1);
                }
            }
            
            /* Grid Overlay */
            .grid-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-image: 
                    linear-gradient(rgba(0, 212, 255, 0.02) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(0, 212, 255, 0.02) 1px, transparent 1px);
                background-size: 50px 50px;
                pointer-events: none;
                z-index: 0;
                animation: gridPulse 8s ease-in-out infinite;
            }
            
            @keyframes gridPulse {
                0%, 100% { opacity: 0.3; }
                50% { opacity: 0.6; }
            }
            
            /* Enhanced Orbs with glow trails */
            .orb-4 {
                width: 180px;
                height: 180px;
                background: linear-gradient(135deg, #39ff14, #22c55e);
                bottom: 30%;
                left: 5%;
                animation-delay: -15s;
            }
            
            .orb-5 {
                width: 120px;
                height: 120px;
                background: linear-gradient(135deg, #feca57, #f97316);
                top: 70%;
                right: 30%;
                animation-delay: -20s;
            }
        </style>
        
        <!-- Animated Background Elements -->
        <div class="login-bg">
            <div class="orb orb-1"></div>
            <div class="orb orb-2"></div>
            <div class="orb orb-3"></div>
            <div class="orb orb-4"></div>
            <div class="orb orb-5"></div>
        </div>
        
        <!-- Floating Particles -->
        <div class="floating-particles">
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
        </div>
        
        <!-- Cyber Scanline -->
        <div class="cyber-scanline"></div>
        
        <!-- Pulse Rings -->
        <div class="pulse-rings">
            <div class="pulse-ring"></div>
            <div class="pulse-ring"></div>
            <div class="pulse-ring"></div>
            <div class="pulse-ring"></div>
        </div>
        
        <!-- Grid Overlay -->
        <div class="grid-overlay"></div>
        """, unsafe_allow_html=True)
        
        # Create centered login container
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Login card with enhanced styling - include styles inline to ensure rendering
            st.markdown('''
            <style>
                .login-card-container {
                    background: linear-gradient(135deg, rgba(2, 2, 8, 0.99) 0%, rgba(5, 5, 15, 0.97) 100%);
                    backdrop-filter: blur(30px);
                    border-radius: 28px;
                    padding: 3.5rem;
                    margin-top: 3rem;
                    box-shadow: 
                        0 30px 100px rgba(0, 0, 0, 0.85),
                        0 0 60px rgba(0, 212, 255, 0.1),
                        inset 0 1px 0 rgba(255, 255, 255, 0.03);
                    border: 1px solid rgba(0, 212, 255, 0.15);
                    position: relative;
                    overflow: hidden;
                    text-align: center;
                }
                .login-card-container::before {
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    height: 4px;
                    background: linear-gradient(90deg, #00d4ff, #8b5cf6, #ff2d92, #ff6b35, #00d4ff);
                    background-size: 400% 100%;
                    animation: gradientFlowCard 6s ease infinite;
                }
                @keyframes gradientFlowCard {
                    0% { background-position: 0% 50%; }
                    50% { background-position: 100% 50%; }
                    100% { background-position: 0% 50%; }
                }
                .card-logo {
                    font-size: 5.5rem;
                    margin-bottom: 1.5rem;
                    animation: logoFloatCard 4s ease-in-out infinite;
                    filter: drop-shadow(0 0 35px rgba(0, 212, 255, 0.6));
                }
                @keyframes logoFloatCard {
                    0%, 100% { transform: translateY(0); }
                    50% { transform: translateY(-12px); }
                }
                .card-title {
                    font-family: 'Orbitron', sans-serif;
                    font-size: 3rem;
                    font-weight: 900;
                    background: linear-gradient(135deg, #00d4ff 0%, #8b5cf6 50%, #ff2d92 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    margin-bottom: 0.8rem;
                    letter-spacing: 3px;
                }
                .card-subtitle {
                    color: #8a8a9a;
                    font-size: 1.1rem;
                    margin-bottom: 1.5rem;
                }
                .card-subtitle span {
                    color: #00d4ff;
                    font-weight: 500;
                }
                .feature-pills {
                    display: flex;
                    justify-content: center;
                    gap: 0.8rem;
                    flex-wrap: wrap;
                    margin-bottom: 1rem;
                }
                .feature-pill {
                    background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
                    border: 1px solid rgba(0, 212, 255, 0.25);
                    padding: 0.5rem 1rem;
                    border-radius: 50px;
                    font-size: 0.85rem;
                    color: #a0a0b0;
                    display: inline-block;
                }
            </style>
            <div class="login-card-container">
                <div class="card-logo">🤖</div>
                <h1 class="card-title">QueryMancer</h1>
                <p class="card-subtitle">
                    🔐 Secure Login to Your <span>AI SQL Assistant</span>
                </p>
                <div class="feature-pills">
                    <span class="feature-pill">🧠 Local AI</span>
                    <span class="feature-pill">⚡ Fast Queries</span>
                    <span class="feature-pill">🔒 100% Private</span>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Show error if exists
            if st.session_state.get('login_error', ''):
                st.error(f"❌ {st.session_state.login_error}")
            
            # Login form
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input(
                    "👤 Username",
                    placeholder="Enter your username",
                    key="login_username"
                )
                
                password = st.text_input(
                    "🔑 Password",
                    type="password",
                    placeholder="Enter your password",
                    key="login_password"
                )
                
                st.markdown("")
                
                submit = st.form_submit_button(
                    "🚀 Login to QueryMancer",
                    use_container_width=True
                )
                
                if submit:
                    # Simple authentication (can be replaced with real auth)
                    if username == "admin" and password == "admin123":
                        st.session_state.authenticated = True
                        st.session_state.logged_in_user = username
                        st.session_state.login_error = ""
                        st.session_state.current_user = username
                        st.rerun()
                    elif username and password:
                        st.session_state.login_error = "Invalid username or password. Try admin/admin123"
                        st.rerun()
                    else:
                        st.session_state.login_error = "Please enter username and password"
                        st.rerun()
            
            # Demo hint using native Streamlit
            st.info("💡 Demo credentials: **admin** / **admin123**")
        
        return False

    def render_welcome_section(self):
        """Render welcome section with features and instructions"""
        if not st.session_state.show_welcome:
            return
        
        # Inject CSS animations
        st.markdown("""<style>
        @keyframes fadeSlideIn { 0% { opacity: 0; transform: translateY(-20px); } 100% { opacity: 1; transform: translateY(0); } }
        @keyframes gradientShift { 0%, 100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
        @keyframes floatBounce { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
        @keyframes glowPulse { 0%, 100% { box-shadow: 0 0 15px rgba(0,212,255,0.15); } 50% { box-shadow: 0 0 35px rgba(139,92,246,0.3); } }
        @keyframes borderGlow { 
            0%, 100% { border-color: rgba(0, 212, 255, 0.25); } 
            33% { border-color: rgba(139, 92, 246, 0.3); } 
            66% { border-color: rgba(255, 45, 146, 0.25); } 
        }
        @keyframes dashboardParticleFloat {
            0% { transform: translateY(100vh) scale(0) rotate(0deg); opacity: 0; }
            10% { opacity: 0.8; }
            90% { opacity: 0.8; }
            100% { transform: translateY(-100vh) scale(1.5) rotate(720deg); opacity: 0; }
        }
        @keyframes dashboardScan {
            0% { top: -4px; opacity: 0; }
            10% { opacity: 0.6; }
            90% { opacity: 0.6; }
            100% { top: 100%; opacity: 0; }
        }
        @keyframes dashboardRingExpand {
            0% { transform: scale(1); opacity: 0.5; border-color: rgba(0, 212, 255, 0.4); }
            50% { border-color: rgba(139, 92, 246, 0.3); }
            100% { transform: scale(6); opacity: 0; border-color: rgba(255, 45, 146, 0.1); }
        }
        @keyframes dashboardOrbDrift {
            0%, 100% { transform: translate(0, 0) scale(1); }
            25% { transform: translate(40px, -30px) scale(1.1); }
            50% { transform: translate(-25px, 20px) scale(0.9); }
            75% { transform: translate(30px, 25px) scale(1.05); }
        }
        @keyframes cardEnter3D {
            0% { opacity: 0; transform: translateY(40px) scale(0.85) rotateX(-10deg); filter: blur(5px); }
            60% { transform: translateY(-5px) scale(1.02) rotateX(2deg); }
            100% { opacity: 1; transform: translateY(0) scale(1) rotateX(0deg); filter: blur(0); }
        }
        .dashboard-particle {
            position: fixed;
            width: 5px;
            height: 5px;
            border-radius: 50%;
            animation: dashboardParticleFloat 15s infinite linear;
            pointer-events: none;
            z-index: 0;
        }
        .dashboard-particle:nth-child(1) { background: #00d4ff; left: 8%; animation-delay: 0s; animation-duration: 14s; }
        .dashboard-particle:nth-child(2) { background: #8b5cf6; left: 18%; animation-delay: -2s; animation-duration: 16s; }
        .dashboard-particle:nth-child(3) { background: #ff2d92; left: 28%; animation-delay: -4s; animation-duration: 18s; }
        .dashboard-particle:nth-child(4) { background: #39ff14; left: 38%; animation-delay: -6s; animation-duration: 15s; }
        .dashboard-particle:nth-child(5) { background: #feca57; left: 48%; animation-delay: -8s; animation-duration: 17s; }
        .dashboard-particle:nth-child(6) { background: #00f5ff; left: 58%; animation-delay: -10s; animation-duration: 19s; }
        .dashboard-particle:nth-child(7) { background: #ff6b6b; left: 68%; animation-delay: -12s; animation-duration: 16s; }
        .dashboard-particle:nth-child(8) { background: #4ecdc4; left: 78%; animation-delay: -14s; animation-duration: 18s; }
        .dashboard-particle:nth-child(9) { background: #a855f7; left: 88%; animation-delay: -3s; animation-duration: 15s; }
        .dashboard-particle:nth-child(10) { background: #14b8a6; left: 95%; animation-delay: -7s; animation-duration: 17s; }
        .dashboard-orb {
            position: fixed;
            border-radius: 50%;
            filter: blur(60px);
            opacity: 0.2;
            animation: dashboardOrbDrift 20s ease-in-out infinite;
            pointer-events: none;
            z-index: 0;
        }
        .dashboard-orb-1 { width: 300px; height: 300px; background: linear-gradient(135deg, #00d4ff, #0099cc); top: 10%; left: 5%; }
        .dashboard-orb-2 { width: 250px; height: 250px; background: linear-gradient(135deg, #ff2d92, #cc0066); bottom: 15%; right: 5%; animation-delay: -7s; }
        .dashboard-orb-3 { width: 200px; height: 200px; background: linear-gradient(135deg, #8b5cf6, #6b21a8); top: 50%; right: 15%; animation-delay: -12s; }
        .dashboard-scanline {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.6), transparent);
            animation: dashboardScan 6s linear infinite;
            pointer-events: none;
            z-index: 1;
        }
        .dashboard-ring {
            position: fixed;
            top: 50%;
            left: 50%;
            width: 80px;
            height: 80px;
            margin: -40px 0 0 -40px;
            border-radius: 50%;
            border: 2px solid rgba(0, 212, 255, 0.3);
            animation: dashboardRingExpand 5s ease-out infinite;
            pointer-events: none;
            z-index: 0;
        }
        .feature-card-animated {
            animation: cardEnter3D 0.7s cubic-bezier(0.34, 1.56, 0.64, 1) backwards;
            transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        }
        .feature-card-animated:hover {
            transform: translateY(-10px) scale(1.03) !important;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.4), 0 0 40px rgba(0, 212, 255, 0.2) !important;
            border-color: rgba(0, 212, 255, 0.5) !important;
        }
        </style>""", unsafe_allow_html=True)
        
        # Render background animations in a separate markdown call (no HTML comments)
        st.markdown("""<div style="position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:0;overflow:hidden;">
            <div class="dashboard-orb dashboard-orb-1"></div>
            <div class="dashboard-orb dashboard-orb-2"></div>
            <div class="dashboard-orb dashboard-orb-3"></div>
            <div class="dashboard-scanline"></div>
            <div class="dashboard-ring" style="animation-delay:0s;"></div>
            <div class="dashboard-ring" style="animation-delay:1.25s;"></div>
            <div class="dashboard-ring" style="animation-delay:2.5s;"></div>
            <div class="dashboard-ring" style="animation-delay:3.75s;"></div>
            <div class="dashboard-particle"></div>
            <div class="dashboard-particle"></div>
            <div class="dashboard-particle"></div>
            <div class="dashboard-particle"></div>
            <div class="dashboard-particle"></div>
            <div class="dashboard-particle"></div>
            <div class="dashboard-particle"></div>
            <div class="dashboard-particle"></div>
            <div class="dashboard-particle"></div>
            <div class="dashboard-particle"></div>
        </div>""", unsafe_allow_html=True)
        
        # Welcome container using native Streamlit container
        with st.container():
            # Welcome header with enhanced animations
            st.markdown("""<div style="background: linear-gradient(135deg, rgba(2,2,8,0.98) 0%, rgba(5,5,15,0.95) 100%); border: 1px solid rgba(0,212,255,0.2); border-radius: 20px; padding: 2rem; text-align: center; margin-bottom: 1rem; animation: fadeSlideIn 0.8s ease-out, glowPulse 4s ease-in-out infinite, borderGlow 6s ease infinite; position: relative; overflow: hidden;"><div style="position: absolute; top: 0; left: 0; right: 0; height: 4px; background: linear-gradient(90deg, #00d4ff, #8b5cf6, #ff2d92, #ff6b35, #00d4ff); background-size: 400% 100%; animation: gradientShift 6s ease infinite;"></div><h2 style="font-size: 2rem; font-weight: 800; background: linear-gradient(135deg, #00d4ff, #8b5cf6, #ff2d92); background-size: 300% 100%; -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; animation: gradientShift 8s ease infinite; margin: 0 0 0.5rem 0;">🤖 Welcome to QueryMancer!</h2><p style="color: #b0b0c5; font-size: 1.05rem; line-height: 1.6; margin: 0;">Your local AI-powered SQL chatbot that converts natural language into precise SQL queries.<br>No data leaves your machine - everything runs locally with <span style="color: #00d4ff; font-weight: 600;">Ollama + Mistral</span>.</p></div>""", unsafe_allow_html=True)
            
            # Feature cards using Streamlit columns
            col1, col2, col3, col4 = st.columns(4)
            
            card_style = "background: linear-gradient(145deg, rgba(2,2,10,0.98) 0%, rgba(8,8,20,0.95) 100%); border: 1px solid rgba(0,212,255,0.15); border-radius: 16px; padding: 1.5rem; text-align: center; height: 180px; transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1); position: relative; overflow: hidden;"
            card_hover_glow = "box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3), 0 0 25px rgba(0, 212, 255, 0.1);"
            card_before = "position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #00d4ff, #8b5cf6, #ff2d92); background-size: 300% 100%; animation: gradientShift 4s ease infinite;"
            icon_style = "font-size: 2.5rem; display: block; margin-bottom: 0.5rem; animation: floatBounce 3s ease-in-out infinite; filter: drop-shadow(0 0 15px rgba(0, 212, 255, 0.4));"
            title_style = "color: #ffffff; font-size: 1.05rem; font-weight: 700; margin: 0.5rem 0; text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);"
            desc_style = "color: #8a8aa0; font-size: 0.85rem; line-height: 1.4; margin: 0;"
            
            with col1:
                st.markdown(f'<div class="feature-card-animated" style="{card_style} {card_hover_glow} animation-delay: 0.1s;"><div style="{card_before}"></div><span style="{icon_style}">🔐</span><div style="{title_style}">100% Local</div><p style="{desc_style}">All AI processing happens on your machine. Your data stays private.</p></div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'<div class="feature-card-animated" style="{card_style} {card_hover_glow} animation-delay: 0.2s;"><div style="{card_before}"></div><span style="{icon_style} animation-delay: 0.2s;">🧠</span><div style="{title_style}">Smart AI</div><p style="{desc_style}">Mistral AI via Ollama for highly accurate SQL generation.</p></div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown(f'<div class="feature-card-animated" style="{card_style} {card_hover_glow} animation-delay: 0.3s;"><div style="{card_before}"></div><span style="{icon_style} animation-delay: 0.4s;">⚡</span><div style="{title_style}">Lightning Fast</div><p style="{desc_style}">Optimized local inference for instant query results.</p></div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown(f'<div class="feature-card-animated" style="{card_style} {card_hover_glow} animation-delay: 0.4s;"><div style="{card_before}"></div><span style="{icon_style} animation-delay: 0.6s;">🎯</span><div style="{title_style}">Schema-Aware</div><p style="{desc_style}">Uses your database schema for maximum accuracy.</p></div>', unsafe_allow_html=True)
            
            # Quick tip with enhanced styling
            st.markdown("""<div style="background: linear-gradient(135deg, rgba(57,255,20,0.08) 0%, rgba(0,212,255,0.05) 100%); border: 1px solid rgba(57,255,20,0.25); border-radius: 14px; padding: 1.2rem 1.5rem; margin-top: 1rem; text-align: center; animation: glowPulse 3s ease-in-out infinite; position: relative; overflow: hidden;"><div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, #39ff14, #00d4ff, #39ff14); background-size: 200% 100%; animation: gradientShift 3s ease infinite;"></div><span style="color: #39ff14; font-weight: 700; text-shadow: 0 0 10px rgba(57, 255, 20, 0.4);">💡 Quick Start</span><p style="color: #a0a0b5; font-size: 0.95rem; margin: 0.3rem 0 0 0;">Just type your question below, like: <code style="background: rgba(0,0,0,0.5); padding: 0.3rem 0.6rem; border-radius: 6px; color: #00d4ff; border: 1px solid rgba(0, 212, 255, 0.2);">Show me all customers from New York</code></p></div>""", unsafe_allow_html=True)
        
        # Close welcome button
        if st.button("✕ Close Welcome", key="close_welcome"):
            st.session_state.show_welcome = False
            st.rerun()

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

    def render_thinking_animation(self, message: str = None):
        """Render thinking animation for local AI processing"""
        if not st.session_state.is_thinking:
            return
            
        thinking_msg = message or st.session_state.thinking_message or "🤖 Local AI is processing your request..."
        
        st.markdown(f"""
        <div class="local-thinking">
            <div class="local-thinking-content">
                <div class="local-thinking-icon">🤖</div>
                <div class="local-thinking-text">{thinking_msg}</div>
                <div class="thinking-dots">
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_query_suggestions(self):
        """Render AI Avatar Animation instead of sample queries after login"""
        # Use inline styles for reliable rendering in Streamlit
        # First inject the keyframe animations
        st.markdown("""<style>
        @keyframes qmAvatarFloat {
            0%, 100% { transform: translateY(0) scale(1) rotate(0deg); }
            25% { transform: translateY(-15px) scale(1.05) rotate(3deg); }
            50% { transform: translateY(-8px) scale(1.02) rotate(-2deg); }
            75% { transform: translateY(-12px) scale(1.04) rotate(2deg); }
        }
        @keyframes qmAvatarGlow {
            0% { filter: drop-shadow(0 0 30px rgba(0, 212, 255, 0.8)) drop-shadow(0 0 60px rgba(0, 212, 255, 0.4)); }
            33% { filter: drop-shadow(0 0 40px rgba(139, 92, 246, 0.9)) drop-shadow(0 0 80px rgba(139, 92, 246, 0.5)); }
            66% { filter: drop-shadow(0 0 35px rgba(255, 45, 146, 0.8)) drop-shadow(0 0 70px rgba(255, 45, 146, 0.4)); }
            100% { filter: drop-shadow(0 0 30px rgba(57, 255, 20, 0.7)) drop-shadow(0 0 60px rgba(57, 255, 20, 0.35)); }
        }
        @keyframes qmPulseRing {
            0% { transform: scale(1); opacity: 0.6; }
            100% { transform: scale(2.5); opacity: 0; }
        }
        @keyframes qmParticleFloat {
            0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.8; }
            25% { transform: translate(25px, -35px) scale(1.4); opacity: 1; }
            50% { transform: translate(-20px, 25px) scale(0.7); opacity: 0.5; }
            75% { transform: translate(30px, 20px) scale(1.3); opacity: 0.9; }
        }
        @keyframes qmWelcomeGradient {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        @keyframes qmPillFadeIn {
            0% { opacity: 0; transform: translateY(20px) scale(0.9); }
            100% { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes qmPromptPulse {
            0%, 100% { box-shadow: 0 0 20px rgba(57, 255, 20, 0.1); }
            50% { box-shadow: 0 0 40px rgba(57, 255, 20, 0.25); }
        }
        </style>""", unsafe_allow_html=True)
        
        # Then render the HTML with inline styles
        st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 2rem 1rem; margin: 1rem auto; max-width: 700px; position: relative;">
            <div style="position: relative; width: 200px; height: 200px; display: flex; align-items: center; justify-content: center; margin-bottom: 1.5rem;">
                <div style="position: absolute; width: 120px; height: 120px; border-radius: 50%; border: 2px solid rgba(0, 212, 255, 0.3); animation: qmPulseRing 2s ease-out infinite;"></div>
                <div style="position: absolute; width: 120px; height: 120px; border-radius: 50%; border: 2px solid rgba(139, 92, 246, 0.3); animation: qmPulseRing 2s ease-out infinite 0.5s;"></div>
                <div style="position: absolute; width: 120px; height: 120px; border-radius: 50%; border: 2px solid rgba(255, 45, 146, 0.3); animation: qmPulseRing 2s ease-out infinite 1s;"></div>
                <div style="position: absolute; width: 10px; height: 10px; background: #00d4ff; border-radius: 50%; top: 15%; left: 20%; box-shadow: 0 0 20px #00d4ff; animation: qmParticleFloat 5s ease-in-out infinite;"></div>
                <div style="position: absolute; width: 8px; height: 8px; background: #8b5cf6; border-radius: 50%; top: 25%; right: 15%; box-shadow: 0 0 18px #8b5cf6; animation: qmParticleFloat 6s ease-in-out infinite 1s;"></div>
                <div style="position: absolute; width: 9px; height: 9px; background: #ff2d92; border-radius: 50%; bottom: 25%; left: 12%; box-shadow: 0 0 18px #ff2d92; animation: qmParticleFloat 5.5s ease-in-out infinite 2s;"></div>
                <div style="position: absolute; width: 7px; height: 7px; background: #39ff14; border-radius: 50%; bottom: 18%; right: 18%; box-shadow: 0 0 15px #39ff14; animation: qmParticleFloat 6.5s ease-in-out infinite 0.5s;"></div>
                <div style="font-size: 4rem; animation: qmAvatarFloat 4s ease-in-out infinite, qmAvatarGlow 3s ease-in-out infinite alternate; z-index: 10;">🤖</div>
            </div>
            <div style="text-align: center; position: relative; z-index: 5;">
                <h2 style="font-family: 'Orbitron', sans-serif; font-size: 1.8rem; font-weight: 800; background: linear-gradient(135deg, #00d4ff 0%, #8b5cf6 25%, #ff2d92 50%, #feca57 75%, #39ff14 100%); background-size: 400% 400%; -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; animation: qmWelcomeGradient 8s ease infinite; margin: 0 0 0.6rem 0; letter-spacing: 2px;">AI Ready to Assist</h2>
                <p style="color: #a0a0b5; font-size: 1rem; line-height: 1.5; max-width: 400px; margin: 0 auto;">Your intelligent <span style="background: linear-gradient(90deg, #00d4ff, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 600;">SQL Assistant</span> is powered by local AI.</p>
            </div>
            <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 0.6rem; margin-top: 1.5rem; max-width: 500px;">
                <span style="background: linear-gradient(135deg, rgba(0, 212, 255, 0.12) 0%, rgba(139, 92, 246, 0.12) 100%); border: 1px solid rgba(0, 212, 255, 0.25); padding: 0.45rem 0.9rem; border-radius: 50px; font-size: 0.8rem; color: #c0c0d0; animation: qmPillFadeIn 0.6s ease-out both; animation-delay: 0.1s;">🧠 Natural Language</span>
                <span style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.12) 0%, rgba(255, 45, 146, 0.12) 100%); border: 1px solid rgba(139, 92, 246, 0.25); padding: 0.45rem 0.9rem; border-radius: 50px; font-size: 0.8rem; color: #c0c0d0; animation: qmPillFadeIn 0.6s ease-out both; animation-delay: 0.2s;">⚡ Instant SQL</span>
                <span style="background: linear-gradient(135deg, rgba(255, 45, 146, 0.12) 0%, rgba(254, 202, 87, 0.12) 100%); border: 1px solid rgba(255, 45, 146, 0.25); padding: 0.45rem 0.9rem; border-radius: 50px; font-size: 0.8rem; color: #c0c0d0; animation: qmPillFadeIn 0.6s ease-out both; animation-delay: 0.3s;">🔐 100% Local</span>
                <span style="background: linear-gradient(135deg, rgba(57, 255, 20, 0.12) 0%, rgba(0, 212, 255, 0.12) 100%); border: 1px solid rgba(57, 255, 20, 0.25); padding: 0.45rem 0.9rem; border-radius: 50px; font-size: 0.8rem; color: #c0c0d0; animation: qmPillFadeIn 0.6s ease-out both; animation-delay: 0.4s;">📊 Data Insights</span>
                <span style="background: linear-gradient(135deg, rgba(0, 245, 255, 0.12) 0%, rgba(139, 92, 246, 0.12) 100%); border: 1px solid rgba(0, 245, 255, 0.25); padding: 0.45rem 0.9rem; border-radius: 50px; font-size: 0.8rem; color: #c0c0d0; animation: qmPillFadeIn 0.6s ease-out both; animation-delay: 0.5s;">🎯 Schema-Aware</span>
            </div>
            <div style="margin-top: 1.5rem; padding: 1rem 1.5rem; background: linear-gradient(135deg, rgba(57, 255, 20, 0.06) 0%, rgba(0, 212, 255, 0.04) 100%); border: 1px solid rgba(57, 255, 20, 0.2); border-radius: 12px; text-align: center; animation: qmPromptPulse 3s ease-in-out infinite;">
                <p style="margin: 0; color: #b0b0c5; font-size: 0.9rem;"><span style="color: #39ff14; font-weight: 600;">💡 Ready!</span> Type your question below <code style="background: rgba(0, 0, 0, 0.4); padding: 0.2rem 0.5rem; border-radius: 5px; color: #00d4ff; font-family: monospace; margin-left: 0.3rem; font-size: 0.8rem;">Show me all customers</code></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        return None

    def render_schema_explorer(self, schema_data: Dict = None):
        """Render interactive schema explorer"""
        if not st.session_state.show_schema_explorer or not schema_data:
            return
            
        st.markdown('<div class="schema-explorer">', unsafe_allow_html=True)
        st.markdown("### 📋 Database Schema Explorer")
        
        # Filter tables
        search_term = st.text_input("🔍 Search tables:", key="schema_search")
        
        tables = schema_data.get('tables', {})
        filtered_tables = {
            name: info for name, info in tables.items()
            if not search_term or search_term.lower() in name.lower()
        }
        
        # Render tables
        for table_name, table_info in filtered_tables.items():
            with st.expander(f"📊 {table_name} ({len(table_info.get('columns', []))} columns)"):
                columns = table_info.get('columns', [])
                primary_keys = table_info.get('primary_keys', [])
                foreign_keys = table_info.get('foreign_keys', [])
                
                st.markdown("**Columns:**")
                col_tags = []
                for col in columns:
                    tag_class = "primary-key" if col in primary_keys else "foreign-key" if col in foreign_keys else "column-tag"
                    col_tags.append(f'<span class="{tag_class}">{col}</span>')
                
                st.markdown(f"""
                <div class="table-columns">
                    {' '.join(col_tags)}
                </div>
                """, unsafe_allow_html=True)
                
                if primary_keys:
                    st.markdown(f"**Primary Keys:** {', '.join(primary_keys)}")
                if foreign_keys:
                    st.markdown(f"**Foreign Keys:** {', '.join(foreign_keys)}")
                    
                # Quick query button
                if st.button(f"📊 Query {table_name}", key=f"query_table_{table_name}"):
                    st.session_state.user_input = f"Show me all data from {table_name} table"
                    return f"Show me all data from {table_name} table"
        
        st.markdown('</div>', unsafe_allow_html=True)
        return None

    def render_sql_display(self, sql_query: str, confidence: float = 0.0, execution_time: float = 0.0):
        """Render SQL query with syntax highlighting and metadata"""
        # Create a safe version of the SQL for use in JavaScript
        # Use a simpler approach that avoids backslashes in f-strings
        copy_button_html = '<button class="copy-button" onclick="copyToClipboard()">📋 Copy</button>'
        
        header_html = f"""
        <div class="sql-display">
            <div class="sql-header">
                <span>🔍 Generated SQL Query</span>
                <div>
                    <span>Confidence: {confidence:.1%}</span>
                    <span style="margin-left: 1rem;">Time: {execution_time:.2f}s</span>
                    {copy_button_html}
                </div>
            </div>
        """
        
        content_html = f'<div class="sql-content">{sql_query}</div></div>'
        
        # Add a script to handle the copy functionality
        script_html = """
        <script>
        function copyToClipboard() {
            const sqlContent = document.querySelector('.sql-content').innerText;
            navigator.clipboard.writeText(sqlContent)
                .then(() => console.log('SQL copied to clipboard'))
                .catch(err => console.error('Failed to copy: ', err));
        }
        </script>
        """
        
        # Combine all parts and render
        st.markdown(header_html + content_html + script_html, unsafe_allow_html=True)

    def render_results_table(self, results_df: pd.DataFrame, query_stats: Dict = None):
        """Render query results with enhanced formatting"""
        if results_df is None or results_df.empty:
            st.markdown("""
            <div class="error-container">
                <div class="error-title">📭 No Results Found</div>
                <div class="error-details">The query executed successfully but returned no data.</div>
            </div>
            """, unsafe_allow_html=True)
            return
            
        stats = query_stats or {}
        rows_count = len(results_df)
        cols_count = len(results_df.columns)
        
        st.markdown(f"""
        <div class="results-container">
            <div class="results-header">
                <div class="results-title">📊 Query Results</div>
                <div class="results-stats">
                    {rows_count} rows × {cols_count} columns
                    {f" • {stats.get('execution_time', 0):.3f}s" if stats.get('execution_time') else ""}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Display the dataframe
        st.dataframe(
            results_df,
            width="stretch",
            height=min(400, (rows_count + 1) * 35)
        )
        
        # Additional stats
        if rows_count > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", rows_count)
            with col2:
                st.metric("Columns", cols_count)
            with col3:
                if stats.get('execution_time'):
                    st.metric("Query Time", f"{stats['execution_time']:.3f}s")
        
        st.markdown('</div>', unsafe_allow_html=True)

    def render_error_display(self, error: Exception, sql_query: str = None, suggestions: List[str] = None):
        """Render error information with helpful suggestions"""
        error_type = type(error).__name__
        error_message = str(error)
        
        st.markdown(f"""
        <div class="error-container">
            <div class="error-title">❌ {error_type}</div>
            <div class="error-details">{error_message}</div>
        """, unsafe_allow_html=True)
        
        if sql_query:
            st.markdown("**Generated SQL:**")
            st.code(sql_query, language="sql")
            
        if suggestions:
            st.markdown("""
            <div class="error-suggestions">
                <strong>💡 Suggestions:</strong>
                <ul>
            """, unsafe_allow_html=True)
            for suggestion in suggestions:
                st.markdown(f"<li>{suggestion}</li>", unsafe_allow_html=True)
            st.markdown("</ul></div>", unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

    def render_chat_message(self, message: str, is_user: bool = False, timestamp: datetime = None, 
                          sql_query: str = None, results_df: pd.DataFrame = None, 
                          confidence: float = 0.0, execution_time: float = 0.0):
        """Render a chat message with enhanced styling"""
        message_class = "user-message" if is_user else "local-ai-message"
        timestamp_str = timestamp.strftime("%H:%M:%S") if timestamp else ""
        
        st.markdown(f"""
        <div class="{message_class}">
            <div class="message-content">
                {message}
                {f'<div class="message-timestamp">{timestamp_str}</div>' if timestamp_str else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Render SQL and results for AI messages
        if not is_user:
            if sql_query:
                self.render_sql_display(sql_query, confidence, execution_time)
            if results_df is not None:
                self.render_results_table(results_df, {
                    'execution_time': execution_time,
                    'confidence': confidence
                })

    def render_sidebar(self, schema_data: Dict = None):
        """Render enhanced sidebar with controls and statistics"""
        with st.sidebar:
            # Logo and title with enhanced styling
            st.markdown("""
            <div style="
                text-align: center; 
                padding: 1.5rem 0;
                margin-bottom: 1.5rem;
                background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%);
                border-radius: 16px;
                border: 1px solid rgba(0, 212, 255, 0.2);
            ">
                <div style="font-size: 3rem; margin-bottom: 0.5rem; filter: drop-shadow(0 0 15px rgba(0, 212, 255, 0.5));">🤖</div>
                <h2 style="
                    font-family: 'Orbitron', sans-serif;
                    font-size: 1.3rem;
                    background: linear-gradient(135deg, #00d4ff 0%, #8b5cf6 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin: 0;
                ">Control Panel</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Local AI Status Section
            st.markdown("""
            <div style="
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 2px;
                color: #00d4ff;
                margin-bottom: 0.5rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid rgba(0, 212, 255, 0.2);
            ">📊 System Status</div>
            """, unsafe_allow_html=True)
            
            # Status indicators
            status_items = [
                ("Database", "Connected" if st.session_state.db_connected else "Disconnected", st.session_state.db_connected),
                ("AI Model", "Ready" if st.session_state.mistral_loaded else "Not Loaded", st.session_state.mistral_loaded),
                ("Schema", "Loaded" if st.session_state.schema_loaded else "Loading...", st.session_state.schema_loaded),
            ]
            
            for name, status, is_active in status_items:
                color = "#39ff14" if is_active else "#ff4757"
                st.markdown(f"""
                <div style="
                    display: flex;
                    align-items: center;
                    padding: 0.5rem 0.8rem;
                    margin: 0.3rem 0;
                    background: rgba(0, 0, 0, 0.2);
                    border-radius: 8px;
                    border-left: 3px solid {color};
                ">
                    <span style="
                        width: 8px;
                        height: 8px;
                        background: {color};
                        border-radius: 50%;
                        margin-right: 10px;
                        box-shadow: 0 0 10px {color};
                    "></span>
                    <span style="color: #a0a0b0; font-size: 0.85rem; flex: 1;">{name}</span>
                    <span style="color: {color}; font-size: 0.8rem; font-weight: 500;">{status}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Statistics Section
            st.markdown("""
            <div style="
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 2px;
                color: #00d4ff;
                margin-bottom: 0.5rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid rgba(0, 212, 255, 0.2);
            ">📈 Session Statistics</div>
            """, unsafe_allow_html=True)
            
            total_queries = st.session_state.success_count + st.session_state.error_count
            success_rate = (st.session_state.success_count / max(total_queries, 1)) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Queries", total_queries)
                st.metric("Successful", st.session_state.success_count)
            with col2:
                st.metric("Success Rate", f"{success_rate:.1f}%")
                st.metric("Failed", st.session_state.error_count)
            
            if st.session_state.execution_times:
                avg_time = np.mean(st.session_state.execution_times)
                st.metric("Avg Response", f"{avg_time:.2f}s")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Quick Actions Section
            st.markdown("""
            <div style="
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 2px;
                color: #00d4ff;
                margin-bottom: 0.5rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid rgba(0, 212, 255, 0.2);
            ">⚡ Quick Actions</div>
            """, unsafe_allow_html=True)
            
            if st.button("🔄 Refresh AI Status", key="refresh_status_btn", use_container_width=True):
                self._check_local_ai_components()
                st.rerun()
                
            if st.button("🗑️ Clear Chat History", key="clear_chat_btn", use_container_width=True):
                st.session_state.messages = []
                st.session_state.query_history = []
                st.rerun()
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Settings Section
            st.markdown("""
            <div style="
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 2px;
                color: #00d4ff;
                margin-bottom: 0.5rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid rgba(0, 212, 255, 0.2);
            ">⚙️ Settings</div>
            """, unsafe_allow_html=True)
            
            st.session_state.show_timestamps = st.checkbox(
                "Show Timestamps", 
                value=st.session_state.show_timestamps
            )
            
            st.session_state.show_sql_in_response = st.checkbox(
                "Show Generated SQL", 
                value=st.session_state.show_sql_in_response
            )
            
            # Schema Explorer Toggle
            if schema_data:
                st.session_state.show_schema_explorer = st.checkbox(
                    "Show Schema Explorer",
                    value=st.session_state.show_schema_explorer
                )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Session Info Section
            st.markdown("""
            <div style="
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 2px;
                color: #00d4ff;
                margin-bottom: 0.5rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid rgba(0, 212, 255, 0.2);
            ">👤 Session Info</div>
            """, unsafe_allow_html=True)
            
            session_duration = (CURRENT_DATETIME - st.session_state.start_time).total_seconds()
            
            st.markdown(f"""
            <div style="
                background: rgba(0, 0, 0, 0.2);
                border-radius: 10px;
                padding: 0.8rem;
                font-size: 0.85rem;
            ">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                    <span style="color: #6a6a7a;">User:</span>
                    <span style="color: #e0e0e0;">{st.session_state.current_user}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                    <span style="color: #6a6a7a;">Session:</span>
                    <span style="color: #00d4ff; font-family: monospace; font-size: 0.75rem;">{st.session_state.session_id[:8]}...</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #6a6a7a;">Duration:</span>
                    <span style="color: #e0e0e0;">{int(session_duration // 60)}m {int(session_duration % 60)}s</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    def render_chat_input(self):
        """Render enhanced chat input with suggestions"""
        # Check if AI is ready
        if not st.session_state.local_ai_ready:
            st.warning("⚠️ Local AI is not ready. Please check the component status above.")
            st.stop()
        
        # Chat input
        user_input = st.chat_input(
            "Ask me anything about your database... (e.g., 'Show me total sales for 2024')",
            key="main_chat_input"
        )
        
        return user_input

    def start_thinking(self, message: str = None):
        """Start the thinking animation"""
        st.session_state.is_thinking = True
        st.session_state.thinking_message = message or "🤖 Local AI is processing your request..."
        st.session_state.processing_start_time = time.time()

    def stop_thinking(self):
        """Stop the thinking animation"""
        st.session_state.is_thinking = False
        st.session_state.thinking_message = ""
        if st.session_state.processing_start_time:
            processing_time = time.time() - st.session_state.processing_start_time
            st.session_state.execution_times.append(processing_time)
            st.session_state.processing_start_time = None

    def add_message(self, content: str, is_user: bool = False, sql_query: str = None, 
                   results_df: pd.DataFrame = None, confidence: float = 0.0, 
                   execution_time: float = 0.0):
        """Add a message to the chat history"""
        message = {
            "content": content,
            "is_user": is_user,
            "timestamp": CURRENT_DATETIME,
            "sql_query": sql_query,
            "results_df": results_df,
            "confidence": confidence,
            "execution_time": execution_time
        }
        st.session_state.messages.append(message)
        
        # Update analytics
        if not is_user:
            if results_df is not None and not results_df.empty:
                st.session_state.success_count += 1
                st.session_state.sql_generation_successes += 1
            else:
                st.session_state.error_count += 1
            
            if confidence > 0:
                st.session_state.confidence_scores.append(confidence)

    def render_chat_history(self):
        """Render the complete chat history"""
        if not st.session_state.messages:
            self.render_welcome_section()
            return
        
        # Render messages
        for message in st.session_state.messages:
            timestamp = message.get('timestamp') if st.session_state.show_timestamps else None
            
            self.render_chat_message(
                message=message['content'],
                is_user=message['is_user'],
                timestamp=timestamp,
                sql_query=message.get('sql_query') if st.session_state.show_sql_in_response else None,
                results_df=message.get('results_df'),
                confidence=message.get('confidence', 0.0),
                execution_time=message.get('execution_time', 0.0)
            )

    def render_analytics_dashboard(self):
        """Render analytics dashboard for query performance"""
        if not st.session_state.show_analytics_dashboard:
            return
            
        st.markdown("### 📊 Analytics Dashboard")
        
        # Performance metrics
        if st.session_state.execution_times:
            fig = px.line(
                x=range(len(st.session_state.execution_times)),
                y=st.session_state.execution_times,
                title="Query Response Times",
                labels={'x': 'Query Number', 'y': 'Response Time (s)'}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Confidence scores
        if st.session_state.confidence_scores:
            fig = px.histogram(
                x=st.session_state.confidence_scores,
                title="Confidence Score Distribution",
                labels={'x': 'Confidence Score', 'y': 'Count'}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_footer(self):
        """Render application footer"""
        st.markdown("""
        <div style="
            text-align: center; 
            margin-top: 3rem; 
            padding: 1.5rem;
            background: linear-gradient(135deg, rgba(10, 10, 25, 0.8) 0%, rgba(15, 15, 35, 0.6) 100%);
            border-top: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 16px 16px 0 0;
        ">
            <p style="
                color: #a0a0b0;
                font-size: 0.9rem;
                margin: 0.3rem 0;
            ">
                🤖 <span style="
                    font-family: 'Orbitron', sans-serif;
                    background: linear-gradient(135deg, #00d4ff 0%, #8b5cf6 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-weight: 600;
                ">QueryMancer</span> v3.0 | Powered by 
                <span style="color: #00d4ff;">Ollama</span> + 
                <span style="color: #8b5cf6;">Mistral</span> | 
                Local AI • Secure Database • No External APIs
            </p>
        </div>
        """, unsafe_allow_html=True)

    def update_analytics(self, success: bool, confidence: float = 0.0, execution_time: float = 0.0):
        """Update session analytics"""
        if success:
            st.session_state.success_count += 1
        else:
            st.session_state.error_count += 1
            
        if execution_time > 0:
            st.session_state.execution_times.append(execution_time)
            
        if confidence > 0:
            st.session_state.confidence_scores.append(confidence)

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        total_queries = st.session_state.success_count + st.session_state.error_count
        success_rate = (st.session_state.success_count / max(total_queries, 1)) * 100
        avg_confidence = np.mean(st.session_state.confidence_scores) if st.session_state.confidence_scores else 0
        avg_execution_time = np.mean(st.session_state.execution_times) if st.session_state.execution_times else 0
        
        return {
            'total_queries': total_queries,
            'success_count': st.session_state.success_count,
            'error_count': st.session_state.error_count,
            'success_rate': success_rate,
            'avg_confidence': avg_confidence,
            'avg_execution_time': avg_execution_time,
            'session_duration': (CURRENT_DATETIME - st.session_state.start_time).total_seconds()
        }

# Utility functions for the UI
# Note: The CSS loading function has been moved to the load_css.py module
# and is now called from the _apply_enhanced_css() method in the EnhancedLocalUI class

def format_sql_query(sql: str) -> str:
    """Format SQL query for display"""
    if not sql:
        return ""
    
    # Basic SQL formatting
    keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER JOIN', 'LEFT JOIN', 
                'RIGHT JOIN', 'GROUP BY', 'ORDER BY', 'HAVING', 'UNION', 
                'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']
    
    formatted_sql = sql
    for keyword in keywords:
        formatted_sql = re.sub(
            f'\\b{keyword}\\b', 
            f'{keyword}', 
            formatted_sql, 
            flags=re.IGNORECASE
        )
    
    return formatted_sql

def validate_sql_query(sql: str) -> Tuple[bool, str]:
    """Basic SQL query validation"""
    if not sql or not sql.strip():
        return False, "Empty SQL query"
    
    # Check for basic SQL structure
    sql_lower = sql.lower().strip()
    
    # Check for common SQL injection patterns
    dangerous_patterns = [
        r';\s*(drop|delete|truncate|alter)\s+',
        r';\s*exec\s*\(',
        r';\s*execute\s*\(',
        r'xp_cmdshell',
        r'sp_executesql'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, sql_lower):
            return False, f"Potentially dangerous SQL pattern detected: {pattern}"
    
    # Check for valid SQL start
    valid_starts = ['select', 'with', 'show', 'explain', 'describe', 'desc']
    if not any(sql_lower.startswith(start) for start in valid_starts):
        return False, "Query must start with a valid SQL command (SELECT, WITH, etc.)"
    
    return True, "Valid SQL query"

def create_sample_data_for_testing() -> pd.DataFrame:
    """Create sample data for testing purposes"""
    return pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'Name': ['Ibad', 'Mohsin', 'Alina', 'Ammama', 'Mam daniya'],
        'Department': ['AI', 'AI', 'CS', 'CS', 'Supervisor'],
        'Salary': [50000, 60000, 70000, 55000, 65000],
        'Join_Date': pd.date_range('2020-01-01', periods=5, freq='3M')
    })

# Export the enhanced UI class
__all__ = ['EnhancedLocalUI', 'load_custom_css', 'format_sql_query', 'validate_sql_query', 'create_sample_data_for_testing']

