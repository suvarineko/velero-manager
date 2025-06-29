"""
OAuth Proxy Header Authentication Module.

This module provides functions to extract and validate user information
from OAuth proxy headers in HTTP requests.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import streamlit as st

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Raised when authentication fails due to missing or invalid credentials."""
    pass


class InvalidHeaderError(Exception):
    """Raised when OAuth proxy headers are malformed or invalid."""
    pass


@dataclass
class UserInfo:
    """User information extracted from OAuth proxy headers."""
    
    username: str
    preferred_username: str
    groups: List[str]
    bearer_token: Optional[str] = None
    raw_headers: Dict[str, str] = None
    
    def __post_init__(self):
        """Ensure raw_headers is initialized."""
        if self.raw_headers is None:
            self.raw_headers = {}
    
    def has_group(self, group: str) -> bool:
        """Check if user belongs to a specific group."""
        return group in self.groups
    
    def is_authenticated(self) -> bool:
        """Check if user has valid authentication data."""
        return bool(self.username and self.preferred_username)


def extract_bearer_token(authorization_header: str) -> Optional[str]:
    """
    Extract bearer token from Authorization header.
    
    Args:
        authorization_header: The Authorization header value
        
    Returns:
        Bearer token string or None if not found/invalid
    """
    if not authorization_header:
        return None
        
    # Authorization header format: "Bearer <token>"
    parts = authorization_header.strip().split()
    if len(parts) != 2 or parts[0].lower() != 'bearer':
        logger.warning("Invalid Authorization header format")
        return None
        
    return parts[1]


def parse_groups(groups_header: str) -> List[str]:
    """
    Parse groups from X-Forwarded-Groups header.
    
    Args:
        groups_header: Comma-separated groups string
        
    Returns:
        List of group names
    """
    if not groups_header:
        return []
        
    # Split by comma and clean up whitespace
    groups = [group.strip() for group in groups_header.split(',')]
    # Filter out empty strings
    return [group for group in groups if group]


def validate_headers(headers: Dict[str, Any]) -> None:
    """
    Validate OAuth proxy headers for presence and format.
    
    Args:
        headers: Dictionary of HTTP headers
        
    Raises:
        AuthenticationError: When required authentication headers are missing
        InvalidHeaderError: When headers are present but malformed
    """
    if not headers:
        logger.error("Authentication attempt failed: No headers provided")
        raise AuthenticationError("No headers provided for authentication")
    
    # Check for required headers
    required_headers = ['X-Forwarded-User']
    for header in required_headers:
        if header not in headers:
            logger.error(f"Authentication attempt failed: Missing required header '{header}'")
            raise AuthenticationError(f"Missing required header: {header}")
        
        value = headers[header]
        if not value or not str(value).strip():
            logger.error(f"Authentication attempt failed: Empty value for header '{header}'")
            raise AuthenticationError(f"Empty value for required header: {header}")
    
    # Validate Authorization header format if present
    auth_header = headers.get('Authorization', '')
    if auth_header:
        if not isinstance(auth_header, str):
            logger.error("Authentication attempt failed: Authorization header must be a string")
            raise InvalidHeaderError("Authorization header must be a string")
            
        # Check Bearer token format
        auth_parts = auth_header.strip().split()
        if len(auth_parts) > 0:
            if len(auth_parts) != 2:
                logger.error("Authentication attempt failed: Invalid Authorization header format")
                raise InvalidHeaderError("Authorization header must be in format 'Bearer <token>'")
            
            if auth_parts[0].lower() != 'bearer':
                logger.error("Authentication attempt failed: Authorization header must use Bearer scheme")
                raise InvalidHeaderError("Authorization header must use Bearer authentication scheme")
            
            if not auth_parts[1].strip():
                logger.error("Authentication attempt failed: Empty bearer token")
                raise InvalidHeaderError("Bearer token cannot be empty")
    
    # Validate groups header format if present
    groups_header = headers.get('X-Forwarded-Groups', '')
    if groups_header:
        if not isinstance(groups_header, str):
            logger.error("Authentication attempt failed: X-Forwarded-Groups header must be a string")
            raise InvalidHeaderError("X-Forwarded-Groups header must be a string")
        
        # Check for valid comma-separated format
        try:
            groups = parse_groups(groups_header)
            # Log suspicious group names (e.g., containing special characters)
            for group in groups:
                if any(char in group for char in ['<', '>', '"', "'", '&', ';']):
                    logger.warning(f"Group name contains suspicious characters: {group}")
        except Exception as e:
            logger.error(f"Authentication attempt failed: Invalid groups format: {e}")
            raise InvalidHeaderError(f"Invalid groups format in X-Forwarded-Groups: {e}")
    
    # Validate username format
    username = headers.get('X-Forwarded-User', '')
    if username:
        username = str(username).strip()
        if len(username) > 253:  # RFC 5321 limit for email local part
            logger.error("Authentication attempt failed: Username too long")
            raise InvalidHeaderError("Username exceeds maximum length of 253 characters")
        
        # Check for obviously invalid characters (basic validation)
        if any(char in username for char in ['<', '>', '"', "'", '&', ';', '\n', '\r', '\t', '\x00']):
            logger.error(f"Authentication attempt failed: Username contains invalid characters: {username}")
            raise InvalidHeaderError("Username contains invalid characters")
    
    logger.debug("Header validation passed successfully")


def validate_and_extract_user(headers: Optional[Dict[str, Any]] = None) -> UserInfo:
    """
    Validate headers and extract user information with comprehensive error handling.
    
    Args:
        headers: Optional dictionary of headers for testing purposes
        
    Returns:
        UserInfo object with validated user data
        
    Raises:
        AuthenticationError: When authentication fails due to missing/invalid credentials
        InvalidHeaderError: When headers are malformed
    """
    logger.info("Starting authentication attempt")
    
    try:
        # If headers not provided, try to get from Streamlit context
        if headers is None:
            headers = _get_streamlit_headers()
            
        # Validate headers first
        validate_headers(headers)
        
        # Extract required headers
        username = headers.get('X-Forwarded-User', '').strip()
        preferred_username = headers.get('X-Forwarded-Preferred-Username', '').strip()
        groups_header = headers.get('X-Forwarded-Groups', '')
        authorization = headers.get('Authorization', '')
        
        # Handle preferred_username fallback
        if not preferred_username:
            preferred_username = username
            logger.info("Using username as preferred_username fallback")
        
        # Extract bearer token with validation
        bearer_token = None
        if authorization:
            bearer_token = extract_bearer_token(authorization)
            if not bearer_token:
                logger.warning("Authorization header present but bearer token extraction failed")
        
        # Parse groups with validation
        groups = parse_groups(groups_header)
        
        user_info = UserInfo(
            username=username,
            preferred_username=preferred_username,
            groups=groups,
            bearer_token=bearer_token,
            raw_headers=dict(headers)
        )
        
        logger.info(f"Authentication successful for user: {username}")
        logger.debug(f"User groups: {groups}")
        logger.debug(f"Bearer token present: {bool(bearer_token)}")
        
        return user_info
        
    except (AuthenticationError, InvalidHeaderError) as e:
        logger.error(f"Authentication failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during authentication: {e}")
        raise AuthenticationError(f"Authentication failed due to unexpected error: {e}")


def get_user_from_headers(headers: Optional[Dict[str, Any]] = None) -> Optional[UserInfo]:
    """
    Extract user information from OAuth proxy headers.
    
    This function provides backward compatibility by catching validation errors
    and returning None instead of raising exceptions.
    
    In Streamlit, headers are typically accessed through st.context or request context.
    For development/testing, headers can be passed directly.
    
    Args:
        headers: Optional dictionary of headers for testing purposes
        
    Returns:
        UserInfo object or None if authentication fails
    """
    try:
        return validate_and_extract_user(headers)
    except (AuthenticationError, InvalidHeaderError) as e:
        logger.warning(f"Authentication failed (returning None): {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error extracting user from headers: {e}")
        return None


def _get_streamlit_headers() -> Dict[str, str]:
    """
    Get headers from Streamlit request context.
    
    Attempts to extract real HTTP headers from the Streamlit request context
    in production, with fallbacks for development and testing scenarios.
    
    Returns:
        Dictionary of headers from OAuth proxy or development setup
    """
    # Try to get real headers from Streamlit context (production)
    try:
        if hasattr(st, 'context') and hasattr(st.context, 'headers'):
            headers = st.context.headers
            if headers:
                # Convert StreamlitHeaders to dictionary
                real_headers = headers.to_dict()
                if real_headers:
                    logger.debug(f"Retrieved {len(real_headers)} headers from Streamlit context")
                    return real_headers
    except Exception as e:
        logger.debug(f"Failed to get headers from Streamlit context: {e}")
    
    # For development, check if headers are stored in session state
    if hasattr(st, 'session_state') and 'auth_headers' in st.session_state:
        logger.debug("Using headers from session state (development)")
        return st.session_state.auth_headers
    
    # Development fallback - environment variables
    import os
    if os.getenv('DEV_MODE', '').lower() == 'true':
        dev_headers = {
            'X-Forwarded-User': os.getenv('DEV_USER', ''),
            'X-Forwarded-Preferred-Username': os.getenv('DEV_PREFERRED_USERNAME', ''),
            'X-Forwarded-Groups': os.getenv('DEV_GROUPS', ''),
            'Authorization': f"Bearer {os.getenv('DEV_TOKEN', '')}" if os.getenv('DEV_TOKEN') else ""
        }
        
        # Filter out empty values
        dev_headers = {k: v for k, v in dev_headers.items() if v}
        
        # Only return if we have at least a username
        if dev_headers.get('X-Forwarded-User'):
            logger.debug(f"Using development headers for user: {dev_headers.get('X-Forwarded-User')}")
            return dev_headers
    
    logger.debug("No headers found in any context")
    return {}


def set_dev_headers(username: str, preferred_username: str = None, 
                   groups: List[str] = None, token: str = None) -> None:
    """
    Set development headers for testing purposes.
    
    Args:
        username: User login identifier
        preferred_username: User display name
        groups: List of user groups
        token: Bearer token
    """
    if preferred_username is None:
        preferred_username = username
        
    if groups is None:
        groups = []
        
    headers = {
        'X-Forwarded-User': username,
        'X-Forwarded-Preferred-Username': preferred_username,
        'X-Forwarded-Groups': ','.join(groups),
        'Authorization': f"Bearer {token}" if token else ""
    }
    
    # Store in session state for development
    if hasattr(st, 'session_state'):
        st.session_state.auth_headers = headers
        
    logger.info(f"Set development headers for user: {username}")


def clear_dev_headers() -> None:
    """Clear development headers from session state."""
    if hasattr(st, 'session_state') and 'auth_headers' in st.session_state:
        del st.session_state.auth_headers
        logger.info("Cleared development headers")