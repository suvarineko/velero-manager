"""
Unit tests for auth.middleware module.

Tests authentication decorators, middleware functions, and RBAC functionality.
"""

import pytest
from unittest.mock import patch, Mock, MagicMock
from typing import List, Optional

from auth.middleware import (
    AccessDeniedError,
    require_auth,
    check_authentication,
    require_groups,
    get_user_groups,
    is_user_admin
)
from auth.auth import UserInfo, AuthenticationError


class TestAccessDeniedError:
    """Test the AccessDeniedError exception."""
    
    def test_access_denied_error_creation(self):
        """Test AccessDeniedError creation and inheritance."""
        error = AccessDeniedError("Access denied")
        
        assert isinstance(error, Exception)
        assert str(error) == "Access denied"


class TestRequireAuthDecorator:
    """Test the @require_auth decorator."""
    
    @patch('auth.middleware.get_current_user')
    def test_require_auth_success(self, mock_get_user, sample_user):
        """Test successful authentication with @require_auth."""
        mock_get_user.return_value = sample_user
        
        @require_auth
        def protected_function():
            return "success"
        
        result = protected_function()
        
        assert result == "success"
        mock_get_user.assert_called_once()
    
    @patch('auth.middleware.get_current_user')
    @patch('auth.middleware.authenticate_user')
    def test_require_auth_fallback_to_authenticate(self, mock_auth_user, mock_get_user, sample_user):
        """Test fallback to authenticate_user when no session."""
        mock_get_user.return_value = None
        mock_auth_user.return_value = sample_user
        
        @require_auth
        def protected_function():
            return "success"
        
        result = protected_function()
        
        assert result == "success"
        mock_get_user.assert_called_once()
        mock_auth_user.assert_called_once()
    
    @patch('auth.middleware.get_current_user')
    @patch('auth.middleware.authenticate_user')
    @patch('auth.middleware._handle_authentication_error')
    def test_require_auth_no_user(self, mock_handle_error, mock_auth_user, mock_get_user):
        """Test @require_auth when no user is authenticated."""
        mock_get_user.return_value = None
        mock_auth_user.return_value = None
        
        @require_auth
        def protected_function():
            return "success"
        
        result = protected_function()
        
        assert result is None
        mock_handle_error.assert_called_once()
    
    @patch('auth.middleware.get_current_user')
    def test_require_auth_with_groups_success(self, mock_get_user, admin_user):
        """Test @require_auth with group requirements - success."""
        mock_get_user.return_value = admin_user
        
        @require_auth(groups=["admin"])
        def admin_function():
            return "admin_success"
        
        result = admin_function()
        
        assert result == "admin_success"
    
    @patch('auth.middleware.get_current_user')
    def test_require_auth_with_groups_multiple(self, mock_get_user, admin_user):
        """Test @require_auth with multiple group options."""
        mock_get_user.return_value = admin_user
        
        @require_auth(groups=["admin", "superuser"])
        def admin_function():
            return "admin_success"
        
        result = admin_function()
        
        assert result == "admin_success"
    
    @patch('auth.middleware.get_current_user')
    @patch('auth.middleware._handle_access_denied')
    def test_require_auth_groups_denied(self, mock_handle_denied, mock_get_user, readonly_user):
        """Test @require_auth with group requirements - access denied."""
        mock_get_user.return_value = readonly_user
        
        @require_auth(groups=["admin"])
        def admin_function():
            return "admin_success"
        
        result = admin_function()
        
        assert result is None
        mock_handle_denied.assert_called_once()
    
    @patch('auth.middleware.get_current_user')
    def test_require_auth_string_group(self, mock_get_user, admin_user):
        """Test @require_auth with single string group."""
        mock_get_user.return_value = admin_user
        
        @require_auth(groups="admin")
        def admin_function():
            return "admin_success"
        
        result = admin_function()
        
        assert result == "admin_success"
    
    @patch('auth.middleware.get_current_user')
    def test_require_auth_no_redirect(self, mock_get_user):
        """Test @require_auth with redirect=False raises exceptions."""
        mock_get_user.return_value = None
        
        @require_auth(redirect=False)
        def protected_function():
            return "success"
        
        with pytest.raises(AuthenticationError):
            protected_function()
    
    @patch('auth.middleware.get_current_user')
    def test_require_auth_groups_no_redirect(self, mock_get_user, readonly_user):
        """Test @require_auth with groups and redirect=False raises exceptions."""
        mock_get_user.return_value = readonly_user
        
        @require_auth(groups=["admin"], redirect=False)
        def admin_function():
            return "admin_success"
        
        with pytest.raises(AccessDeniedError):
            admin_function()
    
    @patch('auth.middleware.get_current_user')
    @patch('auth.middleware._handle_authentication_error')
    def test_require_auth_unexpected_error(self, mock_handle_error, mock_get_user):
        """Test @require_auth handling unexpected errors."""
        mock_get_user.side_effect = Exception("Unexpected error")
        
        @require_auth
        def protected_function():
            return "success"
        
        result = protected_function()
        
        assert result is None
        mock_handle_error.assert_called_once()


class TestCheckAuthentication:
    """Test the check_authentication middleware function."""
    
    @patch('auth.middleware.get_current_user')
    def test_check_authentication_success(self, mock_get_user, sample_user):
        """Test successful authentication check."""
        mock_get_user.return_value = sample_user
        
        user = check_authentication()
        
        assert user == sample_user
        mock_get_user.assert_called_once()
    
    @patch('auth.middleware.get_current_user')
    @patch('auth.middleware.authenticate_user')
    def test_check_authentication_fallback(self, mock_auth_user, mock_get_user, sample_user):
        """Test fallback to authenticate_user."""
        mock_get_user.return_value = None
        mock_auth_user.return_value = sample_user
        
        user = check_authentication()
        
        assert user == sample_user
        mock_auth_user.assert_called_once()
    
    @patch('auth.middleware.get_current_user')
    @patch('auth.middleware.authenticate_user')
    @patch('auth.middleware._display_authentication_required')
    def test_check_authentication_no_user(self, mock_display_auth, mock_auth_user, mock_get_user):
        """Test check_authentication when no user found."""
        mock_get_user.return_value = None
        mock_auth_user.return_value = None
        
        user = check_authentication()
        
        assert user is None
        mock_display_auth.assert_called_once()
    
    @patch('auth.middleware.get_current_user')
    def test_check_authentication_with_groups_success(self, mock_get_user, admin_user):
        """Test check_authentication with group requirements - success."""
        mock_get_user.return_value = admin_user
        
        user = check_authentication(required_groups=["admin"])
        
        assert user == admin_user
    
    @patch('auth.middleware.get_current_user')
    @patch('auth.middleware._display_access_denied')
    def test_check_authentication_groups_denied(self, mock_display_denied, mock_get_user, readonly_user):
        """Test check_authentication with group requirements - denied."""
        mock_get_user.return_value = readonly_user
        
        user = check_authentication(required_groups=["admin"])
        
        assert user is None
        mock_display_denied.assert_called_once_with(["admin"])
    
    @patch('auth.middleware.get_current_user')
    def test_check_authentication_groups_string(self, mock_get_user, admin_user):
        """Test check_authentication with string group requirement."""
        mock_get_user.return_value = admin_user
        
        user = check_authentication(required_groups="admin")
        
        assert user == admin_user
    
    @patch('auth.middleware.get_current_user')
    def test_check_authentication_no_error_display(self, mock_get_user):
        """Test check_authentication with show_error=False."""
        mock_get_user.return_value = None
        
        user = check_authentication(show_error=False)
        
        assert user is None
        # No error display functions should be called
    
    @patch('auth.middleware.get_current_user')
    @patch('streamlit.error')
    def test_check_authentication_exception_handling(self, mock_st_error, mock_get_user):
        """Test check_authentication handles exceptions."""
        mock_get_user.side_effect = Exception("Test error")
        
        user = check_authentication()
        
        assert user is None
        mock_st_error.assert_called_once()


class TestRequireGroups:
    """Test the require_groups utility function."""
    
    def test_require_groups_success(self, admin_user):
        """Test require_groups with matching groups."""
        result = require_groups(["admin"], admin_user)
        
        assert result is True
    
    def test_require_groups_multiple_options(self, admin_user):
        """Test require_groups with multiple group options."""
        result = require_groups(["superuser", "admin"], admin_user)
        
        assert result is True
    
    def test_require_groups_string_group(self, admin_user):
        """Test require_groups with string group."""
        result = require_groups("admin", admin_user)
        
        assert result is True
    
    def test_require_groups_no_match(self, readonly_user):
        """Test require_groups with no matching groups."""
        result = require_groups(["admin"], readonly_user)
        
        assert result is False
    
    def test_require_groups_no_user(self):
        """Test require_groups with no user."""
        result = require_groups(["admin"], None)
        
        assert result is False
    
    def test_require_groups_unauthenticated_user(self):
        """Test require_groups with unauthenticated user."""
        invalid_user = UserInfo(username='', preferred_username='', groups=[])
        result = require_groups(["admin"], invalid_user)
        
        assert result is False
    
    @patch('auth.middleware.get_current_user')
    def test_require_groups_current_user(self, mock_get_user, admin_user):
        """Test require_groups using current user from session."""
        mock_get_user.return_value = admin_user
        
        result = require_groups(["admin"])
        
        assert result is True
        mock_get_user.assert_called_once()


class TestGetUserGroups:
    """Test the get_user_groups utility function."""
    
    def test_get_user_groups_with_user(self, sample_user):
        """Test get_user_groups with provided user."""
        groups = get_user_groups(sample_user)
        
        assert groups == ['user', 'admin']
    
    def test_get_user_groups_no_user(self):
        """Test get_user_groups with no user."""
        groups = get_user_groups(None)
        
        assert groups == []
    
    def test_get_user_groups_unauthenticated(self):
        """Test get_user_groups with unauthenticated user."""
        invalid_user = UserInfo(username='', preferred_username='', groups=[])
        groups = get_user_groups(invalid_user)
        
        assert groups == []
    
    @patch('auth.middleware.get_current_user')
    def test_get_user_groups_current_user(self, mock_get_user, sample_user):
        """Test get_user_groups using current user from session."""
        mock_get_user.return_value = sample_user
        
        groups = get_user_groups()
        
        assert groups == ['user', 'admin']
        mock_get_user.assert_called_once()


class TestIsUserAdmin:
    """Test the is_user_admin utility function."""
    
    def test_is_user_admin_true(self, admin_user):
        """Test is_user_admin with admin user."""
        result = is_user_admin(admin_user)
        
        assert result is True
    
    def test_is_user_admin_cluster_admin(self):
        """Test is_user_admin with cluster-admin user."""
        cluster_admin = UserInfo(
            username='cluster-admin',
            preferred_username='Cluster Admin',
            groups=['cluster-admin']
        )
        
        result = is_user_admin(cluster_admin)
        
        assert result is True
    
    def test_is_user_admin_system_admin(self):
        """Test is_user_admin with system:admin user."""
        system_admin = UserInfo(
            username='system-admin',
            preferred_username='System Admin',
            groups=['system:admin']
        )
        
        result = is_user_admin(system_admin)
        
        assert result is True
    
    def test_is_user_admin_false(self, readonly_user):
        """Test is_user_admin with non-admin user."""
        result = is_user_admin(readonly_user)
        
        assert result is False
    
    def test_is_user_admin_no_user(self):
        """Test is_user_admin with no user."""
        result = is_user_admin(None)
        
        assert result is False
    
    @patch('auth.middleware.get_current_user')
    def test_is_user_admin_current_user(self, mock_get_user, admin_user):
        """Test is_user_admin using current user from session."""
        mock_get_user.return_value = admin_user
        
        result = is_user_admin()
        
        assert result is True
        mock_get_user.assert_called_once()


class TestPrivateHelperFunctions:
    """Test private helper functions."""
    
    def test_check_user_groups_success(self, admin_user):
        """Test _check_user_groups with matching groups."""
        from auth.middleware import _check_user_groups
        
        result = _check_user_groups(admin_user, ["admin"])
        
        assert result is True
    
    def test_check_user_groups_multiple_required(self, admin_user):
        """Test _check_user_groups with multiple required groups."""
        from auth.middleware import _check_user_groups
        
        result = _check_user_groups(admin_user, ["admin", "superuser"])
        
        assert result is True  # User has admin, which is in the required list
    
    def test_check_user_groups_no_match(self, readonly_user):
        """Test _check_user_groups with no matching groups."""
        from auth.middleware import _check_user_groups
        
        result = _check_user_groups(readonly_user, ["admin"])
        
        assert result is False
    
    def test_check_user_groups_no_user_groups(self):
        """Test _check_user_groups with user having no groups."""
        from auth.middleware import _check_user_groups
        
        user_no_groups = UserInfo(
            username='user',
            preferred_username='User',
            groups=[]
        )
        
        result = _check_user_groups(user_no_groups, ["admin"])
        
        assert result is False
    
    def test_check_user_groups_empty_required(self, sample_user):
        """Test _check_user_groups with empty required groups."""
        from auth.middleware import _check_user_groups
        
        result = _check_user_groups(sample_user, [])
        
        assert result is False


class TestStreamlitIntegration:
    """Test Streamlit integration functions."""
    
    @patch('streamlit.error')
    @patch('streamlit.info')
    @patch('streamlit.stop')
    def test_handle_authentication_error(self, mock_stop, mock_info, mock_error):
        """Test _handle_authentication_error function."""
        from auth.middleware import _handle_authentication_error
        
        _handle_authentication_error("Test error message")
        
        mock_error.assert_called_once()
        mock_info.assert_called_once()
        mock_stop.assert_called_once()
    
    @patch('streamlit.error')
    @patch('streamlit.info')
    @patch('streamlit.stop')
    def test_handle_access_denied(self, mock_stop, mock_info, mock_error):
        """Test _handle_access_denied function."""
        from auth.middleware import _handle_access_denied
        
        _handle_access_denied("Test access denied message")
        
        mock_error.assert_called_once()
        mock_info.assert_called_once()
        mock_stop.assert_called_once()
    
    @patch('streamlit.error')
    @patch('streamlit.info')
    @patch('streamlit.stop')
    @patch('streamlit.warning')
    @patch('streamlit.text_input')
    @patch('streamlit.button')
    @patch('streamlit.success')
    @patch('streamlit.rerun')
    @patch('streamlit.expander')
    @patch('os.getenv')
    def test_display_authentication_required_dev_mode(self, mock_getenv, mock_expander, 
                                                    mock_rerun, mock_success, mock_button,
                                                    mock_text_input, mock_warning, mock_stop,
                                                    mock_info, mock_error):
        """Test _display_authentication_required in dev mode."""
        from auth.middleware import _display_authentication_required
        
        # Mock dev mode
        mock_getenv.return_value = 'true'
        mock_button.return_value = False  # Button not clicked
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        
        _display_authentication_required()
        
        mock_error.assert_called_once()
        mock_info.assert_called_once()
        mock_stop.assert_called_once()
        mock_warning.assert_called_once()  # Dev mode warning
    
    @patch('streamlit.error')
    @patch('streamlit.info')
    @patch('streamlit.stop')
    @patch('auth.middleware.get_current_user')
    def test_display_access_denied_with_user(self, mock_get_user, mock_stop, mock_info, mock_error, sample_user):
        """Test _display_access_denied with current user info."""
        from auth.middleware import _display_access_denied
        
        mock_get_user.return_value = sample_user
        
        _display_access_denied(["admin"])
        
        mock_error.assert_called_once()
        # Should call info multiple times (for requirements and user info)
        assert mock_info.call_count >= 2
        mock_stop.assert_called_once()


class TestDecoratorFunctionality:
    """Test decorator functionality and edge cases."""
    
    @patch('auth.middleware.get_current_user')
    def test_decorator_preserves_function_metadata(self, mock_get_user, sample_user):
        """Test that decorator preserves function metadata."""
        mock_get_user.return_value = sample_user
        
        @require_auth
        def test_function():
            """Test function docstring."""
            return "test"
        
        assert test_function.__name__ == "test_function"
        assert test_function.__doc__ == "Test function docstring."
    
    @patch('auth.middleware.get_current_user')
    def test_decorator_with_function_arguments(self, mock_get_user, sample_user):
        """Test decorator works with function arguments."""
        mock_get_user.return_value = sample_user
        
        @require_auth
        def test_function(arg1, arg2, kwarg1=None):
            return f"{arg1}-{arg2}-{kwarg1}"
        
        result = test_function("a", "b", kwarg1="c")
        
        assert result == "a-b-c"
    
    @patch('auth.middleware.get_current_user')
    def test_decorator_with_class_methods(self, mock_get_user, sample_user):
        """Test decorator works with class methods."""
        mock_get_user.return_value = sample_user
        
        class TestClass:
            @require_auth
            def test_method(self):
                return "method_result"
        
        test_obj = TestClass()
        result = test_obj.test_method()
        
        assert result == "method_result"


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_require_groups_empty_user_groups(self):
        """Test require_groups with user having empty groups."""
        user_empty_groups = UserInfo(
            username='user',
            preferred_username='User',
            groups=[]
        )
        
        result = require_groups(["admin"], user_empty_groups)
        
        assert result is False
    
    def test_require_groups_none_user_groups(self):
        """Test require_groups with user having None groups."""
        user_none_groups = UserInfo(
            username='user',
            preferred_username='User',
            groups=None
        )
        
        result = require_groups(["admin"], user_none_groups)
        
        assert result is False
    
    @patch('auth.middleware.get_current_user')
    def test_check_authentication_with_invalid_user(self, mock_get_user):
        """Test check_authentication with invalid user object."""
        mock_get_user.return_value = "invalid_user_object"
        
        user = check_authentication()
        
        # Should handle invalid user gracefully
        assert user is None or user == "invalid_user_object"