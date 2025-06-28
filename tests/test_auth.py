"""
Unit tests for auth.auth module.

Tests header extraction, validation, and user information handling.
"""

import pytest
from unittest.mock import patch, Mock
from typing import Dict, Any

from auth.auth import (
    UserInfo,
    AuthenticationError,
    InvalidHeaderError,
    extract_bearer_token,
    parse_groups,
    validate_headers,
    validate_and_extract_user,
    get_user_from_headers,
    set_dev_headers,
    clear_dev_headers
)


class TestUserInfo:
    """Test the UserInfo dataclass."""
    
    def test_user_info_creation(self):
        """Test basic UserInfo creation."""
        user = UserInfo(
            username='testuser',
            preferred_username='Test User',
            groups=['user', 'admin']
        )
        
        assert user.username == 'testuser'
        assert user.preferred_username == 'Test User'
        assert user.groups == ['user', 'admin']
        assert user.bearer_token is None
        assert user.raw_headers == {}
    
    def test_user_info_with_token(self):
        """Test UserInfo with bearer token."""
        user = UserInfo(
            username='testuser',
            preferred_username='Test User',
            groups=['user'],
            bearer_token='test-token',
            raw_headers={'test': 'header'}
        )
        
        assert user.bearer_token == 'test-token'
        assert user.raw_headers == {'test': 'header'}
    
    def test_has_group(self):
        """Test group membership checking."""
        user = UserInfo(
            username='testuser',
            preferred_username='Test User',
            groups=['user', 'admin']
        )
        
        assert user.has_group('user') is True
        assert user.has_group('admin') is True
        assert user.has_group('readonly') is False
        assert user.has_group('nonexistent') is False
    
    def test_is_authenticated(self):
        """Test authentication status checking."""
        # Valid user
        user = UserInfo(
            username='testuser',
            preferred_username='Test User',
            groups=['user']
        )
        assert user.is_authenticated() is True
        
        # Missing username
        user_no_username = UserInfo(
            username='',
            preferred_username='Test User',
            groups=['user']
        )
        assert user_no_username.is_authenticated() is False
        
        # Missing preferred_username
        user_no_preferred = UserInfo(
            username='testuser',
            preferred_username='',
            groups=['user']
        )
        assert user_no_preferred.is_authenticated() is False


class TestExtractBearerToken:
    """Test bearer token extraction."""
    
    def test_valid_bearer_token(self):
        """Test valid bearer token extraction."""
        header = "Bearer abc123xyz"
        token = extract_bearer_token(header)
        assert token == "abc123xyz"
    
    def test_bearer_with_spaces(self):
        """Test bearer token with extra spaces."""
        header = "   Bearer   abc123xyz   "
        token = extract_bearer_token(header)
        assert token == "abc123xyz"
    
    def test_case_insensitive_bearer(self):
        """Test case insensitive bearer keyword."""
        header = "bearer abc123xyz"
        token = extract_bearer_token(header)
        assert token == "abc123xyz"
        
        header = "BEARER abc123xyz"
        token = extract_bearer_token(header)
        assert token == "abc123xyz"
    
    def test_invalid_format(self):
        """Test invalid authorization header formats."""
        # Wrong scheme
        assert extract_bearer_token("Basic abc123") is None
        
        # Missing token
        assert extract_bearer_token("Bearer") is None
        
        # Too many parts
        assert extract_bearer_token("Bearer abc 123") is None
        
        # Empty header
        assert extract_bearer_token("") is None
        assert extract_bearer_token(None) is None


class TestParseGroups:
    """Test group parsing functionality."""
    
    def test_single_group(self):
        """Test parsing single group."""
        groups = parse_groups("admin")
        assert groups == ["admin"]
    
    def test_multiple_groups(self):
        """Test parsing multiple groups."""
        groups = parse_groups("user,admin,readonly")
        assert groups == ["user", "admin", "readonly"]
    
    def test_groups_with_spaces(self):
        """Test parsing groups with spaces."""
        groups = parse_groups("user, admin , readonly")
        assert groups == ["user", "admin", "readonly"]
    
    def test_empty_groups(self):
        """Test parsing empty group strings."""
        assert parse_groups("") == []
        assert parse_groups(None) == []
        assert parse_groups("  ") == []
    
    def test_groups_with_empty_entries(self):
        """Test parsing groups with empty entries."""
        groups = parse_groups("user,,admin,")
        assert groups == ["user", "admin"]


class TestValidateHeaders:
    """Test header validation functionality."""
    
    def test_valid_headers(self, sample_headers):
        """Test validation with valid headers."""
        # Should not raise any exception
        validate_headers(sample_headers)
    
    def test_minimal_valid_headers(self, minimal_headers):
        """Test validation with minimal required headers."""
        # Should not raise any exception
        validate_headers(minimal_headers)
    
    def test_missing_headers(self):
        """Test validation with missing headers."""
        with pytest.raises(AuthenticationError, match="No headers provided"):
            validate_headers(None)
        
        with pytest.raises(AuthenticationError, match="No headers provided"):
            validate_headers({})
    
    def test_missing_required_header(self):
        """Test validation with missing required headers."""
        headers = {'Authorization': 'Bearer token'}
        
        with pytest.raises(AuthenticationError, match="Missing required header: X-Forwarded-User"):
            validate_headers(headers)
    
    def test_empty_required_header(self):
        """Test validation with empty required headers."""
        headers = {'X-Forwarded-User': ''}
        
        with pytest.raises(AuthenticationError, match="Empty value for required header"):
            validate_headers(headers)
        
        headers = {'X-Forwarded-User': '   '}
        with pytest.raises(AuthenticationError, match="Empty value for required header"):
            validate_headers(headers)
    
    def test_invalid_authorization_format(self):
        """Test validation with invalid authorization header."""
        headers = {
            'X-Forwarded-User': 'testuser',
            'Authorization': 'InvalidFormat'
        }
        
        with pytest.raises(InvalidHeaderError, match="Authorization header must be in format"):
            validate_headers(headers)
    
    def test_invalid_authorization_scheme(self):
        """Test validation with wrong authorization scheme."""
        headers = {
            'X-Forwarded-User': 'testuser',
            'Authorization': 'Basic abc123'
        }
        
        with pytest.raises(InvalidHeaderError, match="Authorization header must use Bearer"):
            validate_headers(headers)
    
    def test_empty_bearer_token(self):
        """Test validation with empty bearer token."""
        headers = {
            'X-Forwarded-User': 'testuser',
            'Authorization': 'Bearer '
        }
        
        with pytest.raises(InvalidHeaderError, match="Bearer token cannot be empty"):
            validate_headers(headers)
    
    def test_invalid_groups_format(self):
        """Test validation with invalid groups format."""
        headers = {
            'X-Forwarded-User': 'testuser',
            'X-Forwarded-Groups': 123  # Not a string
        }
        
        with pytest.raises(InvalidHeaderError, match="X-Forwarded-Groups header must be a string"):
            validate_headers(headers)
    
    def test_suspicious_characters_in_groups(self):
        """Test validation with suspicious characters in groups."""
        headers = {
            'X-Forwarded-User': 'testuser',
            'X-Forwarded-Groups': 'admin,<script>alert("xss")</script>'
        }
        
        # Should not raise exception but should log warning
        validate_headers(headers)
    
    def test_username_too_long(self):
        """Test validation with username that's too long."""
        headers = {
            'X-Forwarded-User': 'a' * 254  # Exceeds RFC 5321 limit
        }
        
        with pytest.raises(InvalidHeaderError, match="Username exceeds maximum length"):
            validate_headers(headers)
    
    def test_username_with_invalid_characters(self):
        """Test validation with invalid characters in username."""
        headers = {
            'X-Forwarded-User': 'user<script>'
        }
        
        with pytest.raises(InvalidHeaderError, match="Username contains invalid characters"):
            validate_headers(headers)


class TestValidateAndExtractUser:
    """Test user validation and extraction."""
    
    def test_successful_extraction(self, sample_headers):
        """Test successful user extraction."""
        user = validate_and_extract_user(sample_headers)
        
        assert isinstance(user, UserInfo)
        assert user.username == 'testuser'
        assert user.preferred_username == 'Test User'
        assert user.groups == ['user', 'admin']
        assert user.bearer_token == 'test-token-123'
        assert user.raw_headers == sample_headers
    
    def test_minimal_headers_extraction(self, minimal_headers):
        """Test extraction with minimal headers."""
        user = validate_and_extract_user(minimal_headers)
        
        assert user.username == 'testuser'
        assert user.preferred_username == 'testuser'  # Fallback
        assert user.groups == []
        assert user.bearer_token is None
    
    def test_extraction_with_invalid_headers(self, invalid_headers):
        """Test extraction with invalid headers."""
        with pytest.raises((AuthenticationError, InvalidHeaderError)):
            validate_and_extract_user(invalid_headers)
    
    @patch('auth.auth._get_streamlit_headers')
    def test_extraction_without_provided_headers(self, mock_get_headers, sample_headers):
        """Test extraction when headers not provided (from Streamlit context)."""
        mock_get_headers.return_value = sample_headers
        
        user = validate_and_extract_user()
        
        assert user.username == 'testuser'
        mock_get_headers.assert_called_once()
    
    def test_preferred_username_fallback(self):
        """Test preferred username fallback to username."""
        headers = {
            'X-Forwarded-User': 'testuser',
            # No X-Forwarded-Preferred-Username
        }
        
        user = validate_and_extract_user(headers)
        assert user.preferred_username == 'testuser'


class TestGetUserFromHeaders:
    """Test the backward-compatible user extraction function."""
    
    def test_successful_extraction(self, sample_headers):
        """Test successful extraction returns UserInfo."""
        user = get_user_from_headers(sample_headers)
        
        assert isinstance(user, UserInfo)
        assert user.username == 'testuser'
    
    def test_failed_extraction_returns_none(self, invalid_headers):
        """Test that failed extraction returns None instead of raising."""
        user = get_user_from_headers(invalid_headers)
        
        assert user is None
    
    def test_missing_headers_returns_none(self):
        """Test that missing headers returns None."""
        user = get_user_from_headers({})
        
        assert user is None


class TestDevHeaderFunctions:
    """Test development header utility functions."""
    
    @patch('streamlit.session_state', {})
    def test_set_dev_headers(self, mock_streamlit):
        """Test setting development headers."""
        with patch('auth.auth.st') as mock_st:
            mock_st.session_state = {}
            
            set_dev_headers('testuser', 'Test User', ['admin'], 'token123')
            
            expected_headers = {
                'X-Forwarded-User': 'testuser',
                'X-Forwarded-Preferred-Username': 'Test User',
                'X-Forwarded-Groups': 'admin',
                'Authorization': 'Bearer token123'
            }
            
            assert mock_st.session_state['auth_headers'] == expected_headers
    
    def test_set_dev_headers_with_defaults(self):
        """Test setting development headers with defaults."""
        with patch('auth.auth.st') as mock_st:
            mock_st.session_state = {}
            
            set_dev_headers('testuser')
            
            expected_headers = {
                'X-Forwarded-User': 'testuser',
                'X-Forwarded-Preferred-Username': 'testuser',
                'X-Forwarded-Groups': '',
                'Authorization': ''
            }
            
            assert mock_st.session_state['auth_headers'] == expected_headers
    
    def test_set_dev_headers_with_multiple_groups(self):
        """Test setting development headers with multiple groups."""
        with patch('auth.auth.st') as mock_st:
            mock_st.session_state = {}
            
            set_dev_headers('testuser', groups=['admin', 'user'])
            
            headers = mock_st.session_state['auth_headers']
            assert headers['X-Forwarded-Groups'] == 'admin,user'
    
    def test_clear_dev_headers(self):
        """Test clearing development headers."""
        with patch('auth.auth.st') as mock_st:
            mock_st.session_state = {'auth_headers': {'test': 'value'}}
            
            clear_dev_headers()
            
            assert 'auth_headers' not in mock_st.session_state


class TestStreamlitIntegration:
    """Test Streamlit integration functions."""
    
    @patch('auth.auth.os.getenv')
    def test_get_streamlit_headers_dev_mode(self, mock_getenv, environment_vars):
        """Test getting headers in development mode."""
        # Mock environment variables
        def getenv_side_effect(key, default=''):
            return environment_vars.get(key, default)
        
        mock_getenv.side_effect = getenv_side_effect
        
        from auth.auth import _get_streamlit_headers
        
        headers = _get_streamlit_headers()
        
        assert headers['X-Forwarded-User'] == 'dev-user'
        assert headers['X-Forwarded-Preferred-Username'] == 'Dev User'
        assert headers['X-Forwarded-Groups'] == 'dev,user'
        assert headers['Authorization'] == 'Bearer dev-token'
    
    def test_get_streamlit_headers_from_session(self):
        """Test getting headers from session state."""
        from auth.auth import _get_streamlit_headers
        
        test_headers = {'X-Forwarded-User': 'sessionuser'}
        
        with patch('auth.auth.st') as mock_st:
            mock_st.session_state = {'auth_headers': test_headers}
            
            headers = _get_streamlit_headers()
            
            assert headers == test_headers
    
    def test_get_streamlit_headers_no_dev_mode(self):
        """Test getting headers when not in dev mode."""
        from auth.auth import _get_streamlit_headers
        
        with patch('auth.auth.os.getenv', return_value=''), \
             patch('auth.auth.st') as mock_st:
            
            mock_st.session_state = {}
            
            headers = _get_streamlit_headers()
            
            assert headers == {}


class TestErrorConditions:
    """Test various error conditions and edge cases."""
    
    def test_authentication_error_inheritance(self):
        """Test that AuthenticationError is properly inherited."""
        error = AuthenticationError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"
    
    def test_invalid_header_error_inheritance(self):
        """Test that InvalidHeaderError is properly inherited."""
        error = InvalidHeaderError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"
    
    def test_validate_and_extract_user_unexpected_error(self):
        """Test handling of unexpected errors in validation."""
        with patch('auth.auth.validate_headers', side_effect=ValueError("Unexpected")):
            with pytest.raises(AuthenticationError, match="Authentication failed due to unexpected error"):
                validate_and_extract_user({'X-Forwarded-User': 'test'})


class TestSecurityScenarios:
    """Test security-related scenarios."""
    
    def test_xss_prevention_in_groups(self):
        """Test XSS prevention in group names."""
        headers = {
            'X-Forwarded-User': 'testuser',
            'X-Forwarded-Groups': 'admin,<script>alert("xss")</script>,user'
        }
        
        # Should validate successfully but log warning
        validate_headers(headers)
        
        # Should extract user but with suspicious group
        user = validate_and_extract_user(headers)
        assert '<script>alert("xss")</script>' in user.groups
    
    def test_injection_prevention_in_username(self):
        """Test injection prevention in username."""
        headers = {
            'X-Forwarded-User': 'user; DROP TABLE users;'
        }
        
        with pytest.raises(InvalidHeaderError, match="Username contains invalid characters"):
            validate_headers(headers)
    
    def test_long_username_handling(self):
        """Test handling of extremely long usernames."""
        headers = {
            'X-Forwarded-User': 'a' * 1000  # Very long username
        }
        
        with pytest.raises(InvalidHeaderError, match="Username exceeds maximum length"):
            validate_headers(headers)
    
    def test_null_byte_injection(self):
        """Test null byte injection prevention."""
        headers = {
            'X-Forwarded-User': 'user\x00admin'
        }
        
        with pytest.raises(InvalidHeaderError, match="Username contains invalid characters"):
            validate_headers(headers)