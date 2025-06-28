"""
Authentication module for Velero Manager.

This module handles OAuth proxy header authentication and user session management.
"""

from .auth import (
    UserInfo,
    AuthenticationError,
    InvalidHeaderError,
    get_user_from_headers,
    validate_and_extract_user,
    validate_headers,
    extract_bearer_token,
    parse_groups,
    set_dev_headers,
    clear_dev_headers
)

from .session import (
    SessionManager,
    get_session_manager,
    create_session,
    get_current_user,
    clear_session,
    authenticate_user
)

from .middleware import (
    AccessDeniedError,
    require_auth,
    check_authentication,
    require_groups,
    get_user_groups,
    is_user_admin
)

__all__ = [
    # Auth module exports
    'UserInfo',
    'AuthenticationError',
    'InvalidHeaderError',
    'get_user_from_headers',
    'validate_and_extract_user',
    'validate_headers',
    'extract_bearer_token',
    'parse_groups',
    'set_dev_headers',
    'clear_dev_headers',
    # Session module exports
    'SessionManager',
    'get_session_manager',
    'create_session',
    'get_current_user',
    'clear_session',
    'authenticate_user',
    # Middleware module exports
    'AccessDeniedError',
    'require_auth',
    'check_authentication',
    'require_groups',
    'get_user_groups',
    'is_user_admin'
]