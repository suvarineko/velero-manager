"""
Authentication decorators and middleware for Streamlit applications.

This module provides decorators and middleware functions to protect routes
and enforce authentication requirements with role-based access control.
"""

import logging
import functools
from typing import List, Optional, Callable, Any, Union
import streamlit as st

from .session import get_current_user, authenticate_user
from .auth import UserInfo, AuthenticationError

logger = logging.getLogger(__name__)


class AccessDeniedError(Exception):
    """Raised when user is authenticated but lacks required permissions."""
    pass


def require_auth(groups: Optional[Union[str, List[str]]] = None, 
                redirect: bool = True):
    """
    Decorator to require authentication and optionally specific group membership.
    
    This decorator can be used to protect functions that should only be accessible
    to authenticated users, optionally with specific group requirements.
    
    Args:
        groups: Optional group(s) required for access. Can be a string or list of strings.
                If None, only authentication is required.
        redirect: Whether to handle redirect/display error in Streamlit context.
                 If False, raises exceptions instead.
    
    Returns:
        Decorated function that enforces authentication requirements
        
    Example:
        @require_auth
        def protected_function():
            return "Only authenticated users can see this"
            
        @require_auth(groups=["admin"])
        def admin_function():
            return "Only admin users can see this"
            
        @require_auth(groups=["admin", "backup-operator"])
        def backup_function():
            return "Users in admin OR backup-operator groups can see this"
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Check authentication
                user = get_current_user()
                if not user:
                    # Try to authenticate from headers if no session exists
                    user = authenticate_user()
                
                if not user or not user.is_authenticated():
                    logger.warning(f"Unauthenticated access attempt to {func.__name__}")
                    if redirect:
                        _handle_authentication_error("Authentication required to access this resource")
                        return None
                    else:
                        raise AuthenticationError("Authentication required")
                
                # Check group requirements if specified
                if groups:
                    required_groups = groups if isinstance(groups, list) else [groups]
                    if not _check_user_groups(user, required_groups):
                        logger.warning(f"Access denied for user {user.username} to {func.__name__}. "
                                     f"Required groups: {required_groups}, User groups: {user.groups}")
                        if redirect:
                            _handle_access_denied(f"Access denied. Required groups: {', '.join(required_groups)}")
                            return None
                        else:
                            raise AccessDeniedError(f"Access denied. Required groups: {required_groups}")
                
                logger.debug(f"Access granted to user {user.username} for {func.__name__}")
                return func(*args, **kwargs)
                
            except (AuthenticationError, AccessDeniedError) as e:
                if redirect:
                    _handle_authentication_error(str(e))
                    return None
                else:
                    raise
            except Exception as e:
                logger.error(f"Unexpected error in auth decorator for {func.__name__}: {e}")
                if redirect:
                    _handle_authentication_error("An unexpected error occurred during authentication")
                    return None
                else:
                    raise
        
        return wrapper
    return decorator


def check_authentication(required_groups: Optional[Union[str, List[str]]] = None,
                        show_error: bool = True) -> Optional[UserInfo]:
    """
    Middleware function to check authentication status for Streamlit pages.
    
    This function should be called at the beginning of Streamlit pages to verify
    authentication. It handles the authentication flow and displays appropriate
    messages to the user.
    
    Args:
        required_groups: Optional group(s) required for access
        show_error: Whether to display error messages in Streamlit UI
        
    Returns:
        UserInfo object if authentication successful, None otherwise
        
    Example:
        def backup_page():
            user = check_authentication(required_groups=["backup-admin"])
            if not user:
                return  # Error already displayed, stop execution
            
            # Page content for authenticated users
            st.title("Backup Management")
    """
    try:
        # Try to get current user from session
        user = get_current_user()
        
        # If no session, try to authenticate from headers
        if not user:
            logger.debug("No active session found, attempting authentication from headers")
            user = authenticate_user()
        
        # Check if authentication was successful
        if not user or not user.is_authenticated():
            logger.info("Authentication check failed - no valid user")
            if show_error:
                _display_authentication_required()
            return None
        
        # Check group requirements if specified
        if required_groups:
            groups_list = required_groups if isinstance(required_groups, list) else [required_groups]
            if not _check_user_groups(user, groups_list):
                logger.warning(f"Access denied for user {user.username}. "
                             f"Required groups: {groups_list}, User groups: {user.groups}")
                if show_error:
                    _display_access_denied(groups_list)
                return None
        
        logger.debug(f"Authentication check passed for user: {user.username}")
        return user
        
    except Exception as e:
        logger.error(f"Error during authentication check: {e}")
        if show_error:
            st.error("An error occurred during authentication. Please try again.")
        return None


def require_groups(required_groups: Union[str, List[str]], 
                  user: Optional[UserInfo] = None) -> bool:
    """
    Check if a user has the required group membership.
    
    Args:
        required_groups: Group(s) required for access
        user: UserInfo object. If None, gets current user from session
        
    Returns:
        True if user has required groups, False otherwise
    """
    if user is None:
        user = get_current_user()
    
    if not user or not user.is_authenticated():
        return False
    
    groups_list = required_groups if isinstance(required_groups, list) else [required_groups]
    return _check_user_groups(user, groups_list)


def get_user_groups(user: Optional[UserInfo] = None) -> List[str]:
    """
    Get the groups for a user.
    
    Args:
        user: UserInfo object. If None, gets current user from session
        
    Returns:
        List of group names the user belongs to
    """
    if user is None:
        user = get_current_user()
    
    if not user or not user.is_authenticated():
        return []
    
    return user.groups


def is_user_admin(user: Optional[UserInfo] = None) -> bool:
    """
    Check if user has admin privileges.
    
    Args:
        user: UserInfo object. If None, gets current user from session
        
    Returns:
        True if user has admin privileges, False otherwise
    """
    admin_groups = ["admin", "cluster-admin", "system:admin"]
    return require_groups(admin_groups, user)


def _check_user_groups(user: UserInfo, required_groups: List[str]) -> bool:
    """
    Check if user belongs to any of the required groups.
    
    Args:
        user: UserInfo object
        required_groups: List of required group names
        
    Returns:
        True if user belongs to at least one required group, False otherwise
    """
    if not user or not user.groups:
        return False
    
    # Check if user belongs to any of the required groups
    user_groups = set(user.groups)
    required_groups_set = set(required_groups)
    
    return bool(user_groups.intersection(required_groups_set))


def _handle_authentication_error(message: str) -> None:
    """Handle authentication errors in Streamlit context."""
    st.error(f"🔒 {message}")
    st.info("Please ensure you are properly authenticated and try again.")
    st.stop()


def _handle_access_denied(message: str) -> None:
    """Handle access denied errors in Streamlit context."""
    st.error(f"🚫 {message}")
    st.info("Contact your administrator if you believe you should have access to this resource.")
    st.stop()


def _display_authentication_required() -> None:
    """Display authentication required message in Streamlit."""
    st.error("🔒 Authentication Required")
    st.info("This page requires authentication. Please ensure you are logged in through the OAuth proxy.")
    
    # Show debug information in development mode
    import os
    if os.getenv('DEV_MODE', '').lower() == 'true':
        st.warning("**Development Mode**: Set authentication headers using the auth module.")
        
        with st.expander("Debug: Set Development Headers"):
            username = st.text_input("Username", value="dev-user")
            groups = st.text_input("Groups (comma-separated)", value="admin,user")
            token = st.text_input("Bearer Token", value="dev-token-123")
            
            if st.button("Set Auth Headers"):
                from .auth import set_dev_headers
                set_dev_headers(
                    username=username,
                    groups=groups.split(',') if groups else [],
                    token=token
                )
                st.success("Development headers set! Please refresh the page.")
                st.rerun()
    
    st.stop()


def _display_access_denied(required_groups: List[str]) -> None:
    """Display access denied message in Streamlit."""
    st.error("🚫 Access Denied")
    st.info(f"This resource requires membership in one of the following groups: {', '.join(required_groups)}")
    
    # Show current user info for debugging
    user = get_current_user()
    if user:
        st.info(f"Current user: {user.preferred_username} ({user.username})")
        if user.groups:
            st.info(f"Your groups: {', '.join(user.groups)}")
        else:
            st.info("You are not assigned to any groups.")
    
    st.stop()