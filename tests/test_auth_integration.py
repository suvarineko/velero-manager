"""
Integration tests for the complete authentication system.

Tests the full authentication flow including headers, validation, session management,
and middleware working together.
"""

import pytest
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any

from auth import (
    get_user_from_headers,
    validate_and_extract_user,
    create_session,
    get_current_user,
    clear_session,
    authenticate_user,
    require_auth,
    check_authentication,
    require_groups,
    is_user_admin,
    AuthenticationError,
    AccessDeniedError
)


class TestCompleteAuthenticationFlow:
    """Test the complete authentication flow from headers to session."""
    
    def test_complete_auth_flow_success(self, mock_streamlit, sample_headers, mock_datetime):
        """Test complete successful authentication flow."""
        # Step 1: Extract user from headers
        user = get_user_from_headers(sample_headers)
        
        assert user is not None
        assert user.username == 'testuser'
        assert user.preferred_username == 'Test User'
        assert user.groups == ['user', 'admin']
        assert user.bearer_token == 'test-token-123'
        
        # Step 2: Create session
        result = create_session(user)
        
        assert result is True
        
        # Step 3: Retrieve user from session
        session_user = get_current_user()
        
        assert session_user == user
        
        # Step 4: Clear session
        clear_session()
        
        # Step 5: Verify session is cleared
        cleared_user = get_current_user()
        
        assert cleared_user is None
    
    def test_auth_flow_with_invalid_headers(self, mock_streamlit):
        """Test authentication flow with invalid headers."""
        invalid_headers = {
            'X-Forwarded-User': '',  # Empty username
            'Authorization': 'InvalidFormat'
        }
        
        # Should return None for invalid headers
        user = get_user_from_headers(invalid_headers)
        
        assert user is None
        
        # Session should not be created
        session_user = get_current_user()
        
        assert session_user is None
    
    def test_auth_flow_with_strict_validation(self, mock_streamlit):
        """Test authentication flow with strict validation."""
        invalid_headers = {
            'X-Forwarded-User': '',  # Empty username
        }
        
        # Strict validation should raise exception
        with pytest.raises(AuthenticationError):
            validate_and_extract_user(invalid_headers)
    
    @patch('auth.auth._get_streamlit_headers')
    def test_authenticate_user_function(self, mock_get_headers, mock_streamlit, sample_headers, mock_datetime):
        """Test the authenticate_user convenience function."""
        mock_get_headers.return_value = sample_headers
        
        # Should extract user and create session
        user = authenticate_user()
        
        assert user is not None
        assert user.username == 'testuser'
        
        # Should be able to get user from session
        session_user = get_current_user()
        
        assert session_user == user


class TestDecoratorIntegration:
    """Test decorator integration with the complete auth system."""
    
    def test_require_auth_decorator_with_session(self, mock_streamlit, sample_user, mock_datetime):
        """Test @require_auth decorator with existing session."""
        # Create session first
        create_session(sample_user)
        
        @require_auth
        def protected_function():
            return "protected_content"
        
        result = protected_function()
        
        assert result == "protected_content"
    
    def test_require_auth_decorator_with_headers(self, mock_streamlit, sample_headers):
        """Test @require_auth decorator authenticating from headers."""
        with patch('auth.middleware.get_current_user', return_value=None), \
             patch('auth.middleware.authenticate_user') as mock_auth:
            
            # Mock authenticate_user to return user from headers
            user = get_user_from_headers(sample_headers)
            mock_auth.return_value = user
            
            @require_auth
            def protected_function():
                return "protected_content"
            
            result = protected_function()
            
            assert result == "protected_content"
            mock_auth.assert_called_once()
    
    def test_require_auth_with_groups_integration(self, mock_streamlit, admin_user, mock_datetime):
        """Test @require_auth with groups using session."""
        # Create session with admin user
        create_session(admin_user)
        
        @require_auth(groups=["admin"])
        def admin_function():
            return "admin_content"
        
        result = admin_function()
        
        assert result == "admin_content"
    
    def test_require_auth_groups_denied_integration(self, mock_streamlit, readonly_user, mock_datetime):
        """Test @require_auth groups denied with session."""
        # Create session with readonly user
        create_session(readonly_user)
        
        with patch('auth.middleware._handle_access_denied') as mock_handle:
            @require_auth(groups=["admin"])
            def admin_function():
                return "admin_content"
            
            result = admin_function()
            
            assert result is None
            mock_handle.assert_called_once()


class TestMiddlewareIntegration:
    """Test middleware integration with the complete auth system."""
    
    def test_check_authentication_with_session(self, mock_streamlit, sample_user, mock_datetime):
        """Test check_authentication with existing session."""
        # Create session
        create_session(sample_user)
        
        user = check_authentication()
        
        assert user == sample_user
    
    def test_check_authentication_with_headers(self, mock_streamlit, sample_headers):
        """Test check_authentication falling back to headers."""
        with patch('auth.middleware.get_current_user', return_value=None), \
             patch('auth.middleware.authenticate_user') as mock_auth:
            
            # Mock authenticate_user to return user from headers
            user = get_user_from_headers(sample_headers)
            mock_auth.return_value = user
            
            result = check_authentication()
            
            assert result == user
            mock_auth.assert_called_once()
    
    def test_check_authentication_with_groups(self, mock_streamlit, admin_user, mock_datetime):
        """Test check_authentication with group requirements."""
        # Create session with admin user
        create_session(admin_user)
        
        user = check_authentication(required_groups=["admin"])
        
        assert user == admin_user
    
    def test_check_authentication_groups_denied(self, mock_streamlit, readonly_user, mock_datetime):
        """Test check_authentication with denied group access."""
        # Create session with readonly user
        create_session(readonly_user)
        
        with patch('auth.middleware._display_access_denied') as mock_display:
            user = check_authentication(required_groups=["admin"])
            
            assert user is None
            mock_display.assert_called_once()


class TestSessionPersistenceIntegration:
    """Test session persistence across different operations."""
    
    def test_session_persistence_across_requests(self, mock_streamlit, sample_user, mock_datetime):
        """Test that session persists across multiple requests."""
        # Create session
        create_session(sample_user)
        
        # Simulate multiple requests/operations
        for i in range(5):
            user = get_current_user()
            assert user == sample_user
            assert user.username == 'testuser'
    
    def test_session_timeout_integration(self, mock_streamlit, sample_user):
        """Test session timeout behavior."""
        # Create session with short timeout
        with patch('auth.session.SessionManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Mock session that expires
            mock_manager.get_current_user.side_effect = [sample_user, None]  # Valid, then expired
            
            # First call should return user
            user1 = get_current_user()
            assert user1 == sample_user
            
            # Second call should return None (expired)
            user2 = get_current_user()
            assert user2 is None
    
    def test_session_refresh_integration(self, mock_streamlit, sample_user, mock_datetime):
        """Test session refresh functionality."""
        from auth.session import get_session_manager
        
        # Create session
        create_session(sample_user)
        
        # Get session manager and refresh
        manager = get_session_manager()
        result = manager.refresh_session()
        
        assert result is True
        
        # Should still be able to get user
        user = get_current_user()
        assert user == sample_user


class TestUtilityFunctionIntegration:
    """Test utility functions integration with the auth system."""
    
    def test_require_groups_with_session(self, mock_streamlit, admin_user, mock_datetime):
        """Test require_groups function with session."""
        # Create session
        create_session(admin_user)
        
        # Test with current user from session
        result = require_groups(["admin"])
        
        assert result is True
    
    def test_is_user_admin_with_session(self, mock_streamlit, admin_user, mock_datetime):
        """Test is_user_admin function with session."""
        # Create session
        create_session(admin_user)
        
        # Test with current user from session
        result = is_user_admin()
        
        assert result is True
    
    def test_get_user_groups_with_session(self, mock_streamlit, sample_user, mock_datetime):
        """Test get_user_groups function with session."""
        # Create session
        create_session(sample_user)
        
        # Test with current user from session
        from auth.middleware import get_user_groups
        groups = get_user_groups()
        
        assert groups == ['user', 'admin']


class TestErrorHandlingIntegration:
    """Test error handling across the complete auth system."""
    
    def test_authentication_error_propagation(self, mock_streamlit):
        """Test that authentication errors propagate correctly."""
        invalid_headers = {'X-Forwarded-User': ''}
        
        # validate_and_extract_user should raise
        with pytest.raises(AuthenticationError):
            validate_and_extract_user(invalid_headers)
        
        # get_user_from_headers should return None
        user = get_user_from_headers(invalid_headers)
        assert user is None
        
        # authenticate_user should return None
        with patch('auth.session.get_user_from_headers', return_value=None):
            user = authenticate_user()
            assert user is None
    
    def test_decorator_error_handling_integration(self, mock_streamlit):
        """Test decorator error handling in complete flow."""
        # No session, no valid headers
        with patch('auth.middleware.get_current_user', return_value=None), \
             patch('auth.middleware.authenticate_user', return_value=None), \
             patch('auth.middleware._handle_authentication_error') as mock_handle:
            
            @require_auth
            def protected_function():
                return "protected"
            
            result = protected_function()
            
            assert result is None
            mock_handle.assert_called_once()
    
    def test_middleware_error_handling_integration(self, mock_streamlit):
        """Test middleware error handling in complete flow."""
        # Exception during authentication check
        with patch('auth.middleware.get_current_user', side_effect=Exception("Test error")), \
             patch('streamlit.error') as mock_error:
            
            user = check_authentication()
            
            assert user is None
            mock_error.assert_called_once()


class TestDevelopmentModeIntegration:
    """Test development mode features integration."""
    
    def test_dev_headers_integration(self, mock_streamlit, environment_vars):
        """Test development headers integration."""
        from auth.auth import set_dev_headers, clear_dev_headers
        
        # Set development headers
        set_dev_headers('devuser', 'Dev User', ['admin'], 'dev-token')
        
        # Should be able to authenticate
        with patch('auth.auth._get_streamlit_headers') as mock_get_headers:
            expected_headers = {
                'X-Forwarded-User': 'devuser',
                'X-Forwarded-Preferred-Username': 'Dev User',
                'X-Forwarded-Groups': 'admin',
                'Authorization': 'Bearer dev-token'
            }
            mock_get_headers.return_value = expected_headers
            
            user = get_user_from_headers()
            
            assert user is not None
            assert user.username == 'devuser'
            assert user.preferred_username == 'Dev User'
            assert user.groups == ['admin']
            assert user.bearer_token == 'dev-token'
        
        # Clear headers
        clear_dev_headers()
    
    @patch('os.getenv')
    def test_dev_mode_environment_integration(self, mock_getenv, mock_streamlit):
        """Test development mode environment variable integration."""
        # Mock dev mode environment
        def getenv_side_effect(key, default=''):
            env_vars = {
                'DEV_MODE': 'true',
                'DEV_USER': 'env-user',
                'DEV_PREFERRED_USERNAME': 'Env User',
                'DEV_GROUPS': 'user,readonly',
                'DEV_TOKEN': 'env-token'
            }
            return env_vars.get(key, default)
        
        mock_getenv.side_effect = getenv_side_effect
        
        from auth.auth import _get_streamlit_headers
        
        headers = _get_streamlit_headers()
        
        assert headers['X-Forwarded-User'] == 'env-user'
        assert headers['X-Forwarded-Preferred-Username'] == 'Env User'
        assert headers['X-Forwarded-Groups'] == 'user,readonly'
        assert headers['Authorization'] == 'Bearer env-token'


class TestSecurityIntegration:
    """Test security features integration."""
    
    def test_security_validation_integration(self, mock_streamlit):
        """Test security validation across the system."""
        # Test malicious headers
        malicious_headers = {
            'X-Forwarded-User': 'user<script>alert("xss")</script>',
            'X-Forwarded-Groups': 'admin,<script>evil</script>',
            'Authorization': 'Bearer token'
        }
        
        # Should reject due to invalid characters
        user = get_user_from_headers(malicious_headers)
        assert user is None
        
        # Strict validation should raise
        with pytest.raises((AuthenticationError, Exception)):
            validate_and_extract_user(malicious_headers)
    
    def test_session_security_integration(self, mock_streamlit, sample_user, mock_datetime):
        """Test session security features."""
        # Create session
        create_session(sample_user)
        
        # Get session info
        from auth.session import get_session_manager
        manager = get_session_manager()
        info = manager.get_session_info()
        
        # Should contain security-relevant info
        assert 'session_id' in info
        assert 'created_at' in info
        assert 'expires_at' in info
        assert 'username' in info
        
        # Session ID should be properly formatted
        assert sample_user.username in info['session_id']


class TestPerformanceIntegration:
    """Test performance-related integration scenarios."""
    
    def test_multiple_auth_operations_performance(self, mock_streamlit, sample_user, mock_datetime):
        """Test multiple authentication operations."""
        # Create session once
        create_session(sample_user)
        
        # Multiple operations should reuse session
        for i in range(10):
            user = get_current_user()
            assert user == sample_user
            
            # Utility functions should work efficiently
            assert require_groups(['user'], user) is True
            assert is_user_admin(user) is True  # sample_user has admin group
    
    def test_session_manager_singleton_integration(self, mock_streamlit):
        """Test session manager singleton behavior."""
        from auth.session import get_session_manager
        
        # Multiple calls should return same instance
        manager1 = get_session_manager()
        manager2 = get_session_manager()
        manager3 = get_session_manager()
        
        assert manager1 is manager2
        assert manager2 is manager3