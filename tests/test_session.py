"""
Unit tests for auth.session module.

Tests session management functionality including creation, retrieval, and expiration.
"""

import pytest
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any

from auth.session import (
    SessionManager,
    get_session_manager,
    create_session,
    get_current_user,
    clear_session,
    authenticate_user
)
from auth.auth import UserInfo


class TestSessionManager:
    """Test the SessionManager class."""
    
    def test_session_manager_initialization(self, mock_streamlit):
        """Test SessionManager initialization."""
        manager = SessionManager(session_timeout_minutes=60)
        
        assert manager.session_timeout == timedelta(minutes=60)
    
    def test_default_timeout(self, mock_streamlit):
        """Test default session timeout."""
        manager = SessionManager()
        
        assert manager.session_timeout == timedelta(minutes=480)  # 8 hours
    
    def test_session_manager_without_streamlit(self):
        """Test SessionManager fails without Streamlit."""
        with patch('auth.session.st', None):
            with pytest.raises(RuntimeError, match="SessionManager requires Streamlit session_state"):
                SessionManager()


class TestCreateSession:
    """Test session creation functionality."""
    
    def test_create_session_success(self, mock_streamlit, sample_user, mock_datetime):
        """Test successful session creation."""
        manager = SessionManager()
        
        result = manager.create_session(sample_user)
        
        assert result is True
        
        # Verify session data was stored
        session_state = mock_streamlit['session_state']
        session_state.__setitem__.assert_called()
        
        # Check the calls made to session_state
        calls = session_state.__setitem__.call_args_list
        call_keys = [call[0][0] for call in calls]
        
        assert SessionManager.USER_KEY in call_keys
        assert SessionManager.SESSION_ID_KEY in call_keys
        assert SessionManager.SESSION_CREATED_KEY in call_keys
        assert SessionManager.SESSION_LAST_ACCESSED_KEY in call_keys
        assert SessionManager.SESSION_EXPIRES_KEY in call_keys
    
    def test_create_session_with_invalid_user(self, mock_streamlit):
        """Test session creation with invalid user."""
        manager = SessionManager()
        
        # Test with None user
        result = manager.create_session(None)
        assert result is False
        
        # Test with unauthenticated user
        invalid_user = UserInfo(username='', preferred_username='', groups=[])
        result = manager.create_session(invalid_user)
        assert result is False
    
    def test_create_session_with_exception(self, mock_streamlit, sample_user):
        """Test session creation with exception."""
        manager = SessionManager()
        
        # Mock session_state to raise exception
        mock_streamlit['session_state'].__setitem__.side_effect = Exception("Test error")
        
        result = manager.create_session(sample_user)
        assert result is False
    
    def test_session_id_generation(self, mock_streamlit, sample_user, mock_datetime):
        """Test session ID generation."""
        manager = SessionManager()
        
        with patch('auth.session.datetime') as mock_dt:
            mock_dt.utcnow.return_value = datetime(2023, 1, 1, 12, 0, 0)
            mock_dt.utcnow.return_value.timestamp.return_value = 1672574400.0
            
            manager.create_session(sample_user)
            
            # Check that session ID includes username and timestamp
            calls = mock_streamlit['session_state'].__setitem__.call_args_list
            session_id_call = next(call for call in calls if call[0][0] == SessionManager.SESSION_ID_KEY)
            session_id = session_id_call[0][1]
            
            assert 'testuser_' in session_id


class TestGetCurrentUser:
    """Test current user retrieval functionality."""
    
    def test_get_current_user_success(self, mock_streamlit, sample_user, mock_datetime):
        """Test successful user retrieval."""
        manager = SessionManager()
        
        # Mock session state with valid session
        session_data = {
            SessionManager.USER_KEY: sample_user,
            SessionManager.SESSION_EXPIRES_KEY: datetime.utcnow() + timedelta(hours=1)
        }
        
        def mock_contains(key):
            return key in session_data
        
        def mock_getitem(key):
            if key in session_data:
                return session_data[key]
            raise KeyError(key)
        
        mock_streamlit['session_state'].__contains__.side_effect = mock_contains
        mock_streamlit['session_state'].__getitem__.side_effect = mock_getitem
        
        user = manager.get_current_user()
        
        assert user == sample_user
        
        # Verify last accessed time was updated
        mock_streamlit['session_state'].__setitem__.assert_called_with(
            SessionManager.SESSION_LAST_ACCESSED_KEY, mock_datetime.utcnow.return_value
        )
    
    def test_get_current_user_no_session(self, mock_streamlit):
        """Test user retrieval when no session exists."""
        manager = SessionManager()
        
        mock_streamlit['session_state'].__contains__.return_value = False
        
        user = manager.get_current_user()
        
        assert user is None
    
    def test_get_current_user_expired_session(self, mock_streamlit, sample_user):
        """Test user retrieval with expired session."""
        manager = SessionManager()
        
        # Mock session state with expired session
        session_data = {
            SessionManager.USER_KEY: sample_user,
            SessionManager.SESSION_EXPIRES_KEY: datetime.utcnow() - timedelta(hours=1)  # Expired
        }
        
        def mock_contains(key):
            return key in session_data
        
        def mock_getitem(key):
            if key in session_data:
                return session_data[key]
            raise KeyError(key)
        
        mock_streamlit['session_state'].__contains__.side_effect = mock_contains
        mock_streamlit['session_state'].__getitem__.side_effect = mock_getitem
        
        user = manager.get_current_user()
        
        assert user is None
        
        # Verify session was cleared
        mock_streamlit['session_state'].__delitem__.assert_called()
    
    def test_get_current_user_with_exception(self, mock_streamlit):
        """Test user retrieval with exception."""
        manager = SessionManager()
        
        mock_streamlit['session_state'].__contains__.side_effect = Exception("Test error")
        
        user = manager.get_current_user()
        
        assert user is None


class TestClearSession:
    """Test session clearing functionality."""
    
    def test_clear_session_success(self, mock_streamlit, sample_user):
        """Test successful session clearing."""
        manager = SessionManager()
        
        # Mock session state with data
        session_data = {
            SessionManager.USER_KEY: sample_user,
            SessionManager.SESSION_ID_KEY: 'test-session-id',
            SessionManager.SESSION_CREATED_KEY: datetime.utcnow(),
            SessionManager.SESSION_LAST_ACCESSED_KEY: datetime.utcnow(),
            SessionManager.SESSION_EXPIRES_KEY: datetime.utcnow() + timedelta(hours=1)
        }
        
        def mock_contains(key):
            return key in session_data
        
        def mock_getitem(key):
            if key in session_data:
                return session_data[key]
            raise KeyError(key)
        
        def mock_delitem(key):
            if key in session_data:
                del session_data[key]
        
        mock_streamlit['session_state'].__contains__.side_effect = mock_contains
        mock_streamlit['session_state'].__getitem__.side_effect = mock_getitem
        mock_streamlit['session_state'].__delitem__.side_effect = mock_delitem
        
        manager.clear_session()
        
        # Verify all session keys were deleted
        assert len(session_data) == 0
    
    def test_clear_session_with_exception(self, mock_streamlit):
        """Test session clearing with exception."""
        manager = SessionManager()
        
        mock_streamlit['session_state'].__delitem__.side_effect = Exception("Test error")
        
        # Should not raise exception
        manager.clear_session()
    
    def test_clear_session_no_user(self, mock_streamlit):
        """Test clearing session when no user exists."""
        manager = SessionManager()
        
        mock_streamlit['session_state'].__contains__.return_value = False
        
        # Should not raise exception
        manager.clear_session()


class TestRefreshSession:
    """Test session refresh functionality."""
    
    def test_refresh_session_success(self, mock_streamlit, sample_user, mock_datetime):
        """Test successful session refresh."""
        manager = SessionManager()
        
        # Mock existing valid session
        session_data = {
            SessionManager.USER_KEY: sample_user,
            SessionManager.SESSION_EXPIRES_KEY: datetime.utcnow() + timedelta(hours=1)
        }
        
        def mock_contains(key):
            return key in session_data
        
        def mock_getitem(key):
            if key in session_data:
                return session_data[key]
            raise KeyError(key)
        
        def mock_setitem(key, value):
            session_data[key] = value
        
        mock_streamlit['session_state'].__contains__.side_effect = mock_contains
        mock_streamlit['session_state'].__getitem__.side_effect = mock_getitem
        mock_streamlit['session_state'].__setitem__.side_effect = mock_setitem
        
        result = manager.refresh_session()
        
        assert result is True
        
        # Verify expiration time was updated
        new_expires = session_data[SessionManager.SESSION_EXPIRES_KEY]
        expected_expires = mock_datetime.utcnow.return_value + manager.session_timeout
        assert new_expires == expected_expires
    
    def test_refresh_session_no_active_session(self, mock_streamlit):
        """Test refresh when no active session."""
        manager = SessionManager()
        
        mock_streamlit['session_state'].__contains__.return_value = False
        
        result = manager.refresh_session()
        
        assert result is False


class TestIsSessionValid:
    """Test session validation functionality."""
    
    def test_is_session_valid_true(self, mock_streamlit, sample_user):
        """Test session validation returns True for valid session."""
        manager = SessionManager()
        
        # Mock valid session
        session_data = {
            SessionManager.USER_KEY: sample_user,
            SessionManager.SESSION_EXPIRES_KEY: datetime.utcnow() + timedelta(hours=1)
        }
        
        def mock_contains(key):
            return key in session_data
        
        def mock_getitem(key):
            if key in session_data:
                return session_data[key]
            raise KeyError(key)
        
        mock_streamlit['session_state'].__contains__.side_effect = mock_contains
        mock_streamlit['session_state'].__getitem__.side_effect = mock_getitem
        
        assert manager.is_session_valid() is True
    
    def test_is_session_valid_false(self, mock_streamlit):
        """Test session validation returns False for invalid session."""
        manager = SessionManager()
        
        mock_streamlit['session_state'].__contains__.return_value = False
        
        assert manager.is_session_valid() is False


class TestGetSessionInfo:
    """Test session info retrieval."""
    
    def test_get_session_info_success(self, mock_streamlit, sample_user):
        """Test successful session info retrieval."""
        manager = SessionManager()
        
        created_at = datetime.utcnow()
        last_accessed = datetime.utcnow()
        expires_at = datetime.utcnow() + timedelta(hours=1)
        
        session_data = {
            SessionManager.USER_KEY: sample_user,
            SessionManager.SESSION_ID_KEY: 'test-session-id',
            SessionManager.SESSION_CREATED_KEY: created_at,
            SessionManager.SESSION_LAST_ACCESSED_KEY: last_accessed,
            SessionManager.SESSION_EXPIRES_KEY: expires_at
        }
        
        def mock_contains(key):
            return key in session_data
        
        def mock_getitem(key):
            if key in session_data:
                return session_data[key]
            raise KeyError(key)
        
        def mock_get(key, default=None):
            return session_data.get(key, default)
        
        mock_streamlit['session_state'].__contains__.side_effect = mock_contains
        mock_streamlit['session_state'].__getitem__.side_effect = mock_getitem
        mock_streamlit['session_state'].get.side_effect = mock_get
        
        info = manager.get_session_info()
        
        assert info['session_id'] == 'test-session-id'
        assert info['created_at'] == created_at
        assert info['last_accessed'] == last_accessed
        assert info['expires_at'] == expires_at
        assert info['username'] == 'testuser'
    
    def test_get_session_info_no_session(self, mock_streamlit):
        """Test session info when no session exists."""
        manager = SessionManager()
        
        mock_streamlit['session_state'].__contains__.return_value = False
        
        info = manager.get_session_info()
        
        assert info == {}


class TestAuthenticateFromHeaders:
    """Test authentication from headers functionality."""
    
    def test_authenticate_from_headers_existing_session(self, mock_streamlit, sample_user):
        """Test authentication when valid session already exists."""
        manager = SessionManager()
        
        # Mock existing valid session
        session_data = {
            SessionManager.USER_KEY: sample_user,
            SessionManager.SESSION_EXPIRES_KEY: datetime.utcnow() + timedelta(hours=1)
        }
        
        def mock_contains(key):
            return key in session_data
        
        def mock_getitem(key):
            if key in session_data:
                return session_data[key]
            raise KeyError(key)
        
        mock_streamlit['session_state'].__contains__.side_effect = mock_contains
        mock_streamlit['session_state'].__getitem__.side_effect = mock_getitem
        
        user = manager.authenticate_from_headers()
        
        assert user == sample_user
    
    @patch('auth.session.get_user_from_headers')
    def test_authenticate_from_headers_new_session(self, mock_get_user, mock_streamlit, sample_user):
        """Test authentication creating new session."""
        manager = SessionManager()
        
        # No existing session
        mock_streamlit['session_state'].__contains__.return_value = False
        
        # Mock successful header authentication
        mock_get_user.return_value = sample_user
        
        user = manager.authenticate_from_headers({'test': 'headers'})
        
        assert user == sample_user
        mock_get_user.assert_called_once_with({'test': 'headers'})
    
    @patch('auth.session.get_user_from_headers')
    def test_authenticate_from_headers_failed_auth(self, mock_get_user, mock_streamlit):
        """Test authentication failure from headers."""
        manager = SessionManager()
        
        # No existing session
        mock_streamlit['session_state'].__contains__.return_value = False
        
        # Mock failed header authentication
        mock_get_user.return_value = None
        
        user = manager.authenticate_from_headers()
        
        assert user is None


class TestGlobalFunctions:
    """Test global convenience functions."""
    
    def test_get_session_manager_singleton(self):
        """Test that get_session_manager returns singleton."""
        with patch('auth.session.st'):
            manager1 = get_session_manager()
            manager2 = get_session_manager()
            
            assert manager1 is manager2
    
    def test_get_session_manager_custom_timeout(self):
        """Test get_session_manager with custom timeout."""
        with patch('auth.session.st'):
            manager = get_session_manager(session_timeout_minutes=120)
            
            assert manager.session_timeout == timedelta(minutes=120)
    
    @patch('auth.session.get_session_manager')
    def test_create_session_function(self, mock_get_manager, sample_user):
        """Test global create_session function."""
        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager
        mock_manager.create_session.return_value = True
        
        result = create_session(sample_user)
        
        assert result is True
        mock_manager.create_session.assert_called_once_with(sample_user)
    
    @patch('auth.session.get_session_manager')
    def test_get_current_user_function(self, mock_get_manager, sample_user):
        """Test global get_current_user function."""
        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager
        mock_manager.get_current_user.return_value = sample_user
        
        user = get_current_user()
        
        assert user == sample_user
        mock_manager.get_current_user.assert_called_once()
    
    @patch('auth.session.get_session_manager')
    def test_clear_session_function(self, mock_get_manager):
        """Test global clear_session function."""
        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager
        
        clear_session()
        
        mock_manager.clear_session.assert_called_once()
    
    @patch('auth.session.get_session_manager')
    def test_authenticate_user_function(self, mock_get_manager, sample_user):
        """Test global authenticate_user function."""
        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager
        mock_manager.authenticate_from_headers.return_value = sample_user
        
        user = authenticate_user({'test': 'headers'})
        
        assert user == sample_user
        mock_manager.authenticate_from_headers.assert_called_once_with({'test': 'headers'})


class TestSessionTimeout:
    """Test session timeout scenarios."""
    
    def test_session_timeout_configuration(self, mock_streamlit):
        """Test different session timeout configurations."""
        # Test default timeout (8 hours)
        manager_default = SessionManager()
        assert manager_default.session_timeout == timedelta(minutes=480)
        
        # Test custom timeout
        manager_custom = SessionManager(session_timeout_minutes=60)
        assert manager_custom.session_timeout == timedelta(minutes=60)
    
    def test_session_expiration_check(self, mock_streamlit, sample_user):
        """Test session expiration checking."""
        manager = SessionManager()
        
        # Create session with very short timeout
        manager.session_timeout = timedelta(seconds=1)
        
        # Mock session data with past expiration
        session_data = {
            SessionManager.USER_KEY: sample_user,
            SessionManager.SESSION_EXPIRES_KEY: datetime.utcnow() - timedelta(seconds=2)
        }
        
        def mock_contains(key):
            return key in session_data
        
        def mock_getitem(key):
            if key in session_data:
                return session_data[key]
            raise KeyError(key)
        
        mock_streamlit['session_state'].__contains__.side_effect = mock_contains
        mock_streamlit['session_state'].__getitem__.side_effect = mock_getitem
        
        user = manager.get_current_user()
        
        # Should return None due to expiration
        assert user is None
        
        # Should have called clear_session
        mock_streamlit['session_state'].__delitem__.assert_called()


class TestErrorHandling:
    """Test error handling in session management."""
    
    def test_session_creation_error_handling(self, mock_streamlit, sample_user):
        """Test error handling during session creation."""
        manager = SessionManager()
        
        # Mock session_state to raise exception on setitem
        mock_streamlit['session_state'].__setitem__.side_effect = Exception("Storage error")
        
        result = manager.create_session(sample_user)
        
        assert result is False
    
    def test_session_retrieval_error_handling(self, mock_streamlit):
        """Test error handling during session retrieval."""
        manager = SessionManager()
        
        # Mock session_state to raise exception
        mock_streamlit['session_state'].__contains__.side_effect = Exception("Access error")
        
        user = manager.get_current_user()
        
        assert user is None
    
    def test_session_info_error_handling(self, mock_streamlit):
        """Test error handling during session info retrieval."""
        manager = SessionManager()
        
        # Mock to raise exception
        mock_streamlit['session_state'].__contains__.side_effect = Exception("Info error")
        
        info = manager.get_session_info()
        
        assert info == {}