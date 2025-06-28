"""
Session management module for Streamlit authentication.

This module provides session management functionality for storing and retrieving
authenticated user information in Streamlit's session state.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import streamlit as st

from .auth import UserInfo, get_user_from_headers, validate_and_extract_user

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages user sessions in Streamlit's session state.
    
    This class provides methods to create, retrieve, and clear user sessions,
    ensuring persistence across page reloads and proper session lifecycle management.
    """
    
    # Session state keys
    USER_KEY = 'authenticated_user'
    SESSION_ID_KEY = 'session_id'
    SESSION_CREATED_KEY = 'session_created_at'
    SESSION_LAST_ACCESSED_KEY = 'session_last_accessed'
    SESSION_EXPIRES_KEY = 'session_expires_at'
    
    def __init__(self, session_timeout_minutes: int = 480):  # 8 hours default
        """
        Initialize the SessionManager.
        
        Args:
            session_timeout_minutes: Session timeout in minutes (default: 480 = 8 hours)
        """
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self._ensure_session_state()
    
    def _ensure_session_state(self) -> None:
        """Ensure Streamlit session state is available and initialized."""
        if not hasattr(st, 'session_state'):
            logger.error("Streamlit session_state not available")
            raise RuntimeError("SessionManager requires Streamlit session_state")
    
    def create_session(self, user_info: UserInfo) -> bool:
        """
        Create a new user session.
        
        Args:
            user_info: UserInfo object containing authenticated user data
            
        Returns:
            True if session was created successfully, False otherwise
        """
        try:
            if not user_info or not user_info.is_authenticated():
                logger.error("Cannot create session: Invalid or unauthenticated user info")
                return False
            
            now = datetime.utcnow()
            expires_at = now + self.session_timeout
            
            # Generate a simple session ID (could be enhanced with UUID)
            session_id = f"{user_info.username}_{now.timestamp()}"
            
            # Store user information in session state
            st.session_state[self.USER_KEY] = user_info
            st.session_state[self.SESSION_ID_KEY] = session_id
            st.session_state[self.SESSION_CREATED_KEY] = now
            st.session_state[self.SESSION_LAST_ACCESSED_KEY] = now
            st.session_state[self.SESSION_EXPIRES_KEY] = expires_at
            
            logger.info(f"Created session for user: {user_info.username} (session_id: {session_id})")
            logger.debug(f"Session expires at: {expires_at}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return False
    
    def get_current_user(self) -> Optional[UserInfo]:
        """
        Get the current authenticated user from session.
        
        Returns:
            UserInfo object if user is authenticated and session is valid, None otherwise
        """
        try:
            # Check if user exists in session
            if self.USER_KEY not in st.session_state:
                logger.debug("No user found in session state")
                return None
            
            user_info = st.session_state[self.USER_KEY]
            
            # Validate session hasn't expired
            if self.SESSION_EXPIRES_KEY in st.session_state:
                expires_at = st.session_state[self.SESSION_EXPIRES_KEY]
                if datetime.utcnow() > expires_at:
                    logger.warning(f"Session expired for user: {user_info.username}")
                    self.clear_session()
                    return None
            
            # Update last accessed time
            st.session_state[self.SESSION_LAST_ACCESSED_KEY] = datetime.utcnow()
            
            logger.debug(f"Retrieved user from session: {user_info.username}")
            return user_info
            
        except Exception as e:
            logger.error(f"Error retrieving current user: {e}")
            return None
    
    def clear_session(self) -> None:
        """Clear the current user session."""
        try:
            # Get username for logging before clearing
            username = None
            if self.USER_KEY in st.session_state:
                user_info = st.session_state[self.USER_KEY]
                username = user_info.username if hasattr(user_info, 'username') else 'unknown'
            
            # Clear all session-related keys
            keys_to_clear = [
                self.USER_KEY,
                self.SESSION_ID_KEY,
                self.SESSION_CREATED_KEY,
                self.SESSION_LAST_ACCESSED_KEY,
                self.SESSION_EXPIRES_KEY
            ]
            
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            logger.info(f"Cleared session for user: {username}")
            
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
    
    def refresh_session(self) -> bool:
        """
        Refresh the current session by extending its expiration time.
        
        Returns:
            True if session was refreshed successfully, False otherwise
        """
        try:
            user = self.get_current_user()
            if not user:
                logger.warning("Cannot refresh session: No active session found")
                return False
            
            # Extend session expiration
            now = datetime.utcnow()
            st.session_state[self.SESSION_EXPIRES_KEY] = now + self.session_timeout
            st.session_state[self.SESSION_LAST_ACCESSED_KEY] = now
            
            logger.info(f"Refreshed session for user: {user.username}")
            return True
            
        except Exception as e:
            logger.error(f"Error refreshing session: {e}")
            return False
    
    def is_session_valid(self) -> bool:
        """
        Check if the current session is valid (exists and not expired).
        
        Returns:
            True if session is valid, False otherwise
        """
        return self.get_current_user() is not None
    
    def get_session_info(self) -> Dict[str, Any]:
        """
        Get information about the current session.
        
        Returns:
            Dictionary containing session information or empty dict if no session
        """
        try:
            if not self.is_session_valid():
                return {}
            
            info = {
                'session_id': st.session_state.get(self.SESSION_ID_KEY),
                'created_at': st.session_state.get(self.SESSION_CREATED_KEY),
                'last_accessed': st.session_state.get(self.SESSION_LAST_ACCESSED_KEY),
                'expires_at': st.session_state.get(self.SESSION_EXPIRES_KEY),
                'username': None
            }
            
            user = st.session_state.get(self.USER_KEY)
            if user and hasattr(user, 'username'):
                info['username'] = user.username
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting session info: {e}")
            return {}
    
    def authenticate_from_headers(self, headers: Optional[Dict[str, Any]] = None) -> Optional[UserInfo]:
        """
        Authenticate user from headers and create session if successful.
        
        This is a convenience method that combines header extraction and session creation.
        
        Args:
            headers: Optional dictionary of headers for testing purposes
            
        Returns:
            UserInfo object if authentication successful, None otherwise
        """
        try:
            # First check if we already have a valid session
            existing_user = self.get_current_user()
            if existing_user:
                logger.debug("Using existing valid session")
                return existing_user
            
            # Try to authenticate from headers
            user_info = get_user_from_headers(headers)
            if user_info and user_info.is_authenticated():
                # Create new session
                if self.create_session(user_info):
                    return user_info
                else:
                    logger.error("Failed to create session after successful authentication")
                    return None
            
            logger.debug("Authentication from headers failed")
            return None
            
        except Exception as e:
            logger.error(f"Error during authentication from headers: {e}")
            return None


# Global session manager instance
_session_manager = None


def get_session_manager(session_timeout_minutes: int = 480) -> SessionManager:
    """
    Get the global SessionManager instance.
    
    Args:
        session_timeout_minutes: Session timeout in minutes (used only on first call)
        
    Returns:
        SessionManager instance
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(session_timeout_minutes)
    return _session_manager


def create_session(user_info: UserInfo) -> bool:
    """
    Convenience function to create a session using the global session manager.
    
    Args:
        user_info: UserInfo object containing authenticated user data
        
    Returns:
        True if session was created successfully, False otherwise
    """
    return get_session_manager().create_session(user_info)


def get_current_user() -> Optional[UserInfo]:
    """
    Convenience function to get current user using the global session manager.
    
    Returns:
        UserInfo object if user is authenticated and session is valid, None otherwise
    """
    return get_session_manager().get_current_user()


def clear_session() -> None:
    """Convenience function to clear session using the global session manager."""
    get_session_manager().clear_session()


def authenticate_user(headers: Optional[Dict[str, Any]] = None) -> Optional[UserInfo]:
    """
    Convenience function to authenticate user from headers.
    
    Args:
        headers: Optional dictionary of headers for testing purposes
        
    Returns:
        UserInfo object if authentication successful, None otherwise
    """
    return get_session_manager().authenticate_from_headers(headers)