"""
CSS loading utilities for QueryMancer UI
Extracts CSS loading functionality into a separate module for better organization
"""

import logging
import streamlit as st
import re
from pathlib import Path
from typing import Dict, Optional, List, Set

logger = logging.getLogger("querymancer.ui.css")

def load_custom_css(theme_variables: Optional[Dict[str, str]] = None) -> bool:
    """
    Load custom CSS from external style.css and welcome-styles.css files
    
    Args:
        theme_variables: Optional dictionary of theme variables to override
        
    Returns:
        bool: True if CSS was loaded successfully, False otherwise
    """
    # Check if style.css exists and read from it
    css_file_path = Path("style.css")
    welcome_css_path = Path("welcome-styles.css")
    
    css_loaded = False
    
    if css_file_path.exists():
        try:
            # Read the CSS from file
            with open(css_file_path, "r", encoding="utf-8") as f:
                css_content = f.read()
            
            # Also load welcome-styles.css if it exists
            welcome_css_content = ""
            if welcome_css_path.exists():
                with open(welcome_css_path, "r", encoding="utf-8") as f:
                    welcome_css_content = f.read()
            
            # Verify that all required CSS classes are present
            warnings = verify_required_ui_classes(css_content)
            if warnings:
                for warning in warnings:
                    logger.warning(warning)
            
            # If theme variables are provided, generate CSS variable overrides
            variable_css = ""
            if theme_variables:
                variable_css = ":root {\n"
                for var_name, var_value in theme_variables.items():
                    var_css_name = f"--{var_name.replace('_', '-')}"
                    variable_css += f"    {var_css_name}: {var_value};\n"
                variable_css += "}\n"
            
            # Add the CSS to the page
            st.markdown(f"""
            <style>
            /* CSS loaded from external style.css */
            {css_content}
            
            /* CSS loaded from welcome-styles.css */
            {welcome_css_content}
            
            /* Dynamic theme variables */
            {variable_css}
            </style>
            """, unsafe_allow_html=True)
            
            # Log successful CSS loading
            logger.info(f"✅ Custom CSS loaded from {css_file_path}")
            css_loaded = True
        except Exception as e:
            logger.error(f"Error loading custom CSS from {css_file_path}: {e}")
            return False
    else:
        logger.warning(f"⚠️ style.css not found at {css_file_path}")
        return False
    
    return css_loaded

def check_css_classes(css_content: str, required_classes: List[str]) -> Set[str]:
    """
    Checks if all required CSS classes are defined in the provided CSS content
    
    Args:
        css_content: The CSS content to check
        required_classes: List of class names that should be present
        
    Returns:
        Set of missing class names
    """
    # Extract all class definitions from the CSS
    class_pattern = r'\.([a-zA-Z0-9_-]+)\s*{'
    found_classes = set(re.findall(class_pattern, css_content))
    
    # Check which required classes are missing
    missing_classes = set()
    for required_class in required_classes:
        if required_class not in found_classes:
            missing_classes.add(required_class)
    
    return missing_classes

def verify_required_ui_classes(css_content: str) -> List[str]:
    """
    Verifies that essential UI classes for QueryMancer are defined in the CSS file
    
    Args:
        css_content: CSS content to check
        
    Returns:
        List of warnings for missing essential classes
    """
    # Just check for the most essential classes that affect core functionality
    essential_classes = [
        # Core structural classes
        'welcome-section', 'features-grid', 'sql-display',
        'user-message', 'local-ai-message', 'message-content'
    ]
    
    try:
        # Extract all class definitions from the CSS
        class_pattern = r'\.([a-zA-Z0-9_-]+)\s*{'
        found_classes = set(re.findall(class_pattern, css_content))
        
        # Check which required classes are missing
        warnings = []
        missing_classes = []
        for required_class in essential_classes:
            if required_class not in found_classes:
                missing_classes.append(required_class)
        
        # Only warn about missing essential classes
        if missing_classes:
            warnings.append(f"Warning: Missing essential CSS classes: {', '.join(missing_classes)}")
            warnings.append("These classes are required for core UI functionality.")
        
        return warnings
    except Exception as e:
        return [f"Error checking CSS classes: {str(e)}"]

def apply_theme_variables(theme: Dict[str, str]) -> None:
    """
    Apply only theme variables as CSS custom properties
    Used as a fallback when style.css is not available
    
    Args:
        theme: Dictionary of theme variables to apply
    """
    # Generate the CSS for the theme variables
    st.markdown(f"""
    <style>
    /* Theme variables */
    :root {{
        --primary-color: {theme.get('primary', '#00d4ff')};
        --secondary-color: {theme.get('secondary', '#ff6b6b')};
        --accent-color: {theme.get('accent', '#4ecdc4')};
        --success-color: {theme.get('success', '#50fa7b')};
        --warning-color: {theme.get('warning', '#f1fa8c')};
        --error-color: {theme.get('error', '#ff5555')};
        --background-color: {theme.get('background', '#0f0f23')};
        --surface-color: {theme.get('surface', '#1a1a2e')};
        --text-color: {theme.get('text', '#ffffff')};
        --text-secondary: {theme.get('text_secondary', '#94a3b8')};
        --ollama-color: {theme.get('ollama_color', '#8b5cf6')};
        --mistral-color: {theme.get('mistral_color', '#ff7849')};
        --accuracy-excellent: {theme.get('accuracy_excellent', '#22c55e')};
        --accuracy-good: {theme.get('accuracy_good', '#3b82f6')};
        --accuracy-fair: {theme.get('accuracy_fair', '#f59e0b')};
        --accuracy-poor: {theme.get('accuracy_poor', '#ef4444')};
    }}
    </style>
    """, unsafe_allow_html=True)
