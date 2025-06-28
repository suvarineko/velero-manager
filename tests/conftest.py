"""
Pytest configuration and fixtures for Velero Manager tests.
"""

import pytest
import os
import sys
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from auth.auth import UserInfo


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit for testing without actual Streamlit context."""
    with patch('streamlit.session_state') as mock_session_state:
        # Mock session_state as a dictionary-like object
        mock_session_state.__contains__ = Mock(return_value=False)
        mock_session_state.__getitem__ = Mock(side_effect=KeyError)
        mock_session_state.__setitem__ = Mock()
        mock_session_state.__delitem__ = Mock()
        mock_session_state.get = Mock(return_value=None)
        
        # Mock other Streamlit functions
        with patch('streamlit.error') as mock_error, \
             patch('streamlit.info') as mock_info, \
             patch('streamlit.warning') as mock_warning, \
             patch('streamlit.stop') as mock_stop:
            
            yield {
                'session_state': mock_session_state,
                'error': mock_error,
                'info': mock_info,
                'warning': mock_warning,
                'stop': mock_stop
            }


@pytest.fixture
def sample_headers():
    """Sample OAuth proxy headers for testing."""
    return {
        'X-Forwarded-User': 'testuser',
        'X-Forwarded-Preferred-Username': 'Test User',
        'X-Forwarded-Groups': 'user,admin',
        'Authorization': 'Bearer test-token-123'
    }


@pytest.fixture
def minimal_headers():
    """Minimal required headers for testing."""
    return {
        'X-Forwarded-User': 'testuser'
    }


@pytest.fixture
def admin_headers():
    """Headers for admin user testing."""
    return {
        'X-Forwarded-User': 'admin',
        'X-Forwarded-Preferred-Username': 'Administrator',
        'X-Forwarded-Groups': 'admin,cluster-admin',
        'Authorization': 'Bearer admin-token-456'
    }


@pytest.fixture
def invalid_headers():
    """Invalid headers for testing error conditions."""
    return {
        'X-Forwarded-User': '',  # Empty username
        'X-Forwarded-Groups': 'invalid<>group',  # Invalid characters
        'Authorization': 'InvalidFormat'  # Wrong format
    }


@pytest.fixture
def sample_user():
    """Sample UserInfo object for testing."""
    return UserInfo(
        username='testuser',
        preferred_username='Test User',
        groups=['user', 'admin'],
        bearer_token='test-token-123',
        raw_headers={
            'X-Forwarded-User': 'testuser',
            'X-Forwarded-Preferred-Username': 'Test User',
            'X-Forwarded-Groups': 'user,admin',
            'Authorization': 'Bearer test-token-123'
        }
    )


@pytest.fixture
def admin_user():
    """Admin UserInfo object for testing."""
    return UserInfo(
        username='admin',
        preferred_username='Administrator',
        groups=['admin', 'cluster-admin'],
        bearer_token='admin-token-456',
        raw_headers={
            'X-Forwarded-User': 'admin',
            'X-Forwarded-Preferred-Username': 'Administrator',
            'X-Forwarded-Groups': 'admin,cluster-admin',
            'Authorization': 'Bearer admin-token-456'
        }
    )


@pytest.fixture
def readonly_user():
    """Readonly UserInfo object for testing."""
    return UserInfo(
        username='readonly',
        preferred_username='Read Only User',
        groups=['readonly'],
        bearer_token='readonly-token-789'
    )


@pytest.fixture
def mock_datetime():
    """Mock datetime for consistent testing."""
    from datetime import datetime, timedelta
    
    base_time = datetime(2023, 1, 1, 12, 0, 0)
    
    with patch('auth.session.datetime') as mock_dt:
        mock_dt.utcnow.return_value = base_time
        mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        mock_dt.timedelta = timedelta
        yield mock_dt


@pytest.fixture
def environment_vars():
    """Mock environment variables for testing."""
    env_vars = {
        'DEV_MODE': 'true',
        'DEV_USER': 'dev-user',
        'DEV_PREFERRED_USERNAME': 'Dev User',
        'DEV_GROUPS': 'dev,user',
        'DEV_TOKEN': 'dev-token'
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture(autouse=True)
def reset_session_manager():
    """Reset the global session manager between tests."""
    import auth.session
    auth.session._session_manager = None
    yield
    auth.session._session_manager = None