"""
Kubernetes API client module for Velero Manager application.

This module provides a client for interacting with the Kubernetes API using
bearer token authentication. It includes functionality for resource discovery,
namespace operations, and RBAC permission checking.
"""

import logging
import time
import random
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import wraps
from enum import Enum

from kubernetes import client, config
from kubernetes.client.rest import ApiException
import streamlit as st

# Import auth module for UserInfo integration
try:
    from .auth.auth import UserInfo
except ImportError:
    # Fallback for standalone testing
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'auth'))
    from auth import UserInfo


# Custom Exception Classes for different Kubernetes error types
class K8sBaseException(Exception):
    """Base exception for all Kubernetes client errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 operation: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status_code = status_code
        self.operation = operation
        self.context = context or {}
        self.timestamp = time.time()


class K8sAuthenticationError(K8sBaseException):
    """Authentication failed (401)"""
    pass


class K8sAuthorizationError(K8sBaseException):
    """Authorization failed (403)"""
    pass


class K8sNotFoundError(K8sBaseException):
    """Resource not found (404)"""
    pass


class K8sServerError(K8sBaseException):
    """Server error (500+)"""
    pass


class K8sNetworkError(K8sBaseException):
    """Network/connection error"""
    pass


class K8sTimeoutError(K8sBaseException):
    """Request timeout error"""
    pass


class K8sCircuitBreakerError(K8sBaseException):
    """Circuit breaker is open"""
    pass


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance"""
    failure_threshold: int = 5
    timeout: float = 60.0  # seconds
    half_open_max_calls: int = 3
    
    # Internal state
    failure_count: int = field(default=0, init=False)
    last_failure_time: float = field(default=0.0, init=False)
    state: CircuitBreakerState = field(default=CircuitBreakerState.CLOSED, init=False)
    half_open_calls: int = field(default=0, init=False)
    
    def __post_init__(self):
        self._lock = threading.RLock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time >= self.timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    raise K8sCircuitBreakerError("Circuit breaker is open")
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    raise K8sCircuitBreakerError("Circuit breaker is half-open, max calls exceeded")
                self.half_open_calls += 1
        
        try:
            result = func(*args, **kwargs)
            with self._lock:
                # Success - reset failure count
                self.failure_count = 0
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.state = CircuitBreakerState.CLOSED
                    self.half_open_calls = 0
            return result
        except Exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                elif self.state == CircuitBreakerState.HALF_OPEN:
                    self.state = CircuitBreakerState.OPEN
            raise


@dataclass
class K8sClientConfig:
    """Configuration for Kubernetes client"""
    api_server_url: Optional[str] = None
    verify_ssl: bool = True
    connection_timeout: int = 30
    read_timeout: int = 60
    
    # Enhanced retry configuration
    max_retries: int = 3
    base_retry_delay: float = 1.0  # Base delay for exponential backoff
    max_retry_delay: float = 60.0  # Maximum delay between retries
    retry_jitter: bool = True  # Add random jitter to prevent thundering herd
    
    # Circuit breaker configuration
    enable_circuit_breaker: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    circuit_breaker_half_open_max_calls: int = 3
    
    # Connection pooling configuration
    connection_pool_maxsize: int = 10
    connection_pool_block: bool = False
    
    # Resource discovery configuration
    fast_discovery: bool = True
    resource_cache_ttl: int = 300  # 5 minutes default
    namespace_cache_ttl: int = 600  # 10 minutes default
    max_parallel_requests: int = 5
    include_crd_resources: bool = True
    
    # Graceful degradation
    enable_graceful_degradation: bool = True
    critical_operations: Set[str] = field(default_factory=lambda: {
        'authenticate', 'list_namespaces', 'can_i'
    })
    
    # Enhanced logging
    enable_detailed_logging: bool = True
    log_request_context: bool = True


def with_retry_and_circuit_breaker(operation_name: str = None, critical: bool = False):
    """
    Decorator that adds retry logic and circuit breaker protection to methods.
    
    Args:
        operation_name: Name of the operation for logging
        critical: Whether this is a critical operation (affects graceful degradation)
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, 'config'):
                return func(self, *args, **kwargs)
            
            config = self.config
            op_name = operation_name or func.__name__
            
            # Check if circuit breaker is enabled and available
            circuit_breaker = getattr(self, '_circuit_breaker', None) if config.enable_circuit_breaker else None
            
            # Determine if graceful degradation applies
            can_degrade = (config.enable_graceful_degradation and 
                          not critical and 
                          op_name not in config.critical_operations)
            
            def execute_with_retry():
                last_exception = None
                
                for attempt in range(config.max_retries + 1):
                    try:
                        if circuit_breaker:
                            return circuit_breaker.call(func, self, *args, **kwargs)
                        else:
                            return func(self, *args, **kwargs)
                    
                    except K8sCircuitBreakerError:
                        if can_degrade:
                            self._log_degradation(op_name, "Circuit breaker open")
                            return self._get_degraded_response(op_name, *args, **kwargs)
                        raise
                    
                    except Exception as e:
                        last_exception = e
                        
                        # Convert ApiException to custom exceptions
                        if isinstance(e, ApiException):
                            custom_exception = self._convert_api_exception(e, op_name)
                            last_exception = custom_exception
                            
                            # Check if this is a retryable error
                            if not self._is_retryable_error(custom_exception):
                                if can_degrade and not isinstance(custom_exception, (K8sAuthenticationError,)):
                                    self._log_degradation(op_name, f"Non-retryable error: {custom_exception}")
                                    return self._get_degraded_response(op_name, *args, **kwargs)
                                raise custom_exception
                        
                        # Don't retry on the last attempt
                        if attempt == config.max_retries:
                            break
                        
                        # Calculate delay with exponential backoff and jitter
                        delay = min(
                            config.base_retry_delay * (2 ** attempt),
                            config.max_retry_delay
                        )
                        
                        if config.retry_jitter:
                            delay += random.uniform(0, delay * 0.1)  # Add up to 10% jitter
                        
                        self._log_retry_attempt(op_name, attempt + 1, delay, last_exception)
                        time.sleep(delay)
                
                # All retries exhausted
                if can_degrade:
                    self._log_degradation(op_name, f"All retries exhausted: {last_exception}")
                    return self._get_degraded_response(op_name, *args, **kwargs)
                
                raise last_exception
            
            return execute_with_retry()
        
        return wrapper
    return decorator


def _calculate_exponential_backoff(attempt: int, base_delay: float, max_delay: float, 
                                   jitter: bool = True) -> float:
    """Calculate exponential backoff delay with optional jitter"""
    delay = min(base_delay * (2 ** attempt), max_delay)
    
    if jitter:
        delay += random.uniform(0, delay * 0.1)  # Add up to 10% jitter
    
    return delay


# Velero-relevant resource types for backup operations
VELERO_RELEVANT_RESOURCES = {
    # Core API resources
    'v1': [
        'pods', 'services', 'endpoints', 'persistentvolumeclaims', 
        'configmaps', 'secrets', 'serviceaccounts', 'events'
    ],
    # Apps API resources
    'apps/v1': [
        'deployments', 'replicasets', 'statefulsets', 'daemonsets'
    ],
    # Batch API resources
    'batch/v1': ['jobs'],
    'batch/v1beta1': ['cronjobs'],
    # Networking API resources
    'networking.k8s.io/v1': ['ingresses', 'networkpolicies'],
    # RBAC API resources
    'rbac.authorization.k8s.io/v1': ['roles', 'rolebindings'],
    # Storage API resources
    'storage.k8s.io/v1': ['storageclasses'],
    # Extensions (if available)
    'extensions/v1beta1': ['ingresses']
}

# Velero-relevant operations for backup/restore
VELERO_OPERATIONS = {
    'backup': ['list', 'get'],  # Read operations for backup discovery
    'restore': ['list', 'get', 'create', 'patch'],  # Restore may require creation
    'volume_snapshot': ['list', 'get', 'create', 'delete']  # Volume operations
}

# Common verbs for resource access checking
RBAC_VERBS = ['get', 'list', 'create', 'update', 'patch', 'delete', 'watch']


class KubernetesClient:
    """
    Kubernetes API client with bearer token authentication support.
    
    This client provides methods for interacting with the Kubernetes API,
    including resource discovery, namespace operations, and RBAC checking.
    """
    
    def __init__(self, config_obj: Optional[K8sClientConfig] = None):
        """
        Initialize the Kubernetes client.
        
        Args:
            config_obj: Configuration object for the client
        """
        self.config = config_obj or K8sClientConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize client instances as None - will be set during authentication
        self.api_client: Optional[client.ApiClient] = None
        self.core_v1: Optional[client.CoreV1Api] = None
        self.apps_v1: Optional[client.AppsV1Api] = None
        self.rbac_v1: Optional[client.RbacAuthorizationV1Api] = None
        self.auth_v1: Optional[client.AuthorizationV1Api] = None
        self.custom_objects: Optional[client.CustomObjectsApi] = None
        
        # Enhanced caching system
        self._api_resources_cache: Dict[str, Any] = {}
        self._cache_timestamp: float = 0
        
        # Per-namespace resource caching
        self._namespace_resources_cache: Dict[str, Dict[str, Any]] = {}
        self._namespace_cache_timestamps: Dict[str, float] = {}
        
        # Namespace list caching
        self._namespaces_cache: List[Dict[str, Any]] = []
        self._namespaces_cache_timestamp: float = 0
        
        # CRD caching
        self._crd_cache: List[Dict[str, Any]] = []
        self._crd_cache_timestamp: float = 0
        
        # RBAC caching
        self._rbac_cache: Dict[str, bool] = {}
        self._rbac_cache_timestamp: float = 0
        self._user_roles_cache: List[Dict[str, Any]] = []
        self._user_roles_cache_timestamp: float = 0
        
        # Thread safety for caching
        self._cache_lock = threading.RLock()
        
        # User session information
        self._current_user: Optional[UserInfo] = None
        self._session_id: Optional[str] = None
        
        # Initialize circuit breaker if enabled
        if self.config.enable_circuit_breaker:
            self._circuit_breaker = CircuitBreaker(
                failure_threshold=self.config.circuit_breaker_failure_threshold,
                timeout=self.config.circuit_breaker_timeout,
                half_open_max_calls=self.config.circuit_breaker_half_open_max_calls
            )
        else:
            self._circuit_breaker = None
        
        self.logger.info("KubernetesClient initialized")
    
    def authenticate_with_token(self, bearer_token: str, api_server_url: Optional[str] = None) -> bool:
        """
        Authenticate with the Kubernetes API using a bearer token.
        
        Args:
            bearer_token: The bearer token for authentication
            api_server_url: Optional API server URL override
            
        Returns:
            bool: True if authentication successful, False otherwise
        """
        try:
            # Create configuration
            configuration = client.Configuration()
            
            # Set API server URL
            if api_server_url:
                configuration.host = api_server_url
            elif self.config.api_server_url:
                configuration.host = self.config.api_server_url
            else:
                # Try to load in-cluster configuration first (for pods running in Kubernetes)
                try:
                    config.load_incluster_config()
                    # Get the configuration set by load_incluster_config()
                    incluster_config = client.Configuration.get_default_copy()
                    # Only take the host and SSL settings, not the authentication
                    configuration.host = incluster_config.host
                    configuration.ssl_ca_cert = incluster_config.ssl_ca_cert
                    configuration.verify_ssl = incluster_config.verify_ssl
                    self.logger.info("Successfully loaded in-cluster configuration")
                except Exception as incluster_error:
                    self.logger.debug(f"In-cluster config not available: {incluster_error}")
                    # Fallback to kubeconfig for development/external use
                    try:
                        config.load_kube_config()
                        kubeconfig_config = client.Configuration.get_default_copy()
                        configuration.host = kubeconfig_config.host
                        configuration.ssl_ca_cert = kubeconfig_config.ssl_ca_cert
                        configuration.verify_ssl = kubeconfig_config.verify_ssl
                        self.logger.info("Successfully loaded kubeconfig")
                    except Exception as kubeconfig_error:
                        self.logger.error(f"No API server URL provided and unable to load configuration: "
                                        f"in-cluster error: {incluster_error}, kubeconfig error: {kubeconfig_error}")
                        return False
            
            # Set bearer token
            configuration.api_key = {"authorization": f"Bearer {bearer_token}"}
            
            # Set SSL verification
            configuration.verify_ssl = self.config.verify_ssl
            
            # Set timeouts
            configuration.timeout = self.config.connection_timeout
            
            # Set connection pooling (if supported by the client)
            try:
                import urllib3
                urllib3.util.connection.create_connection = self._create_connection_with_timeout
            except ImportError:
                self.logger.debug("urllib3 not available for connection pooling configuration")
            
            # Configure retry strategy at HTTP level (if supported)
            if hasattr(configuration, 'retries'):
                configuration.retries = False  # Disable built-in retries, we handle them ourselves
            
            # Create API client
            self.api_client = client.ApiClient(configuration)
            
            # Initialize API instances
            self.core_v1 = client.CoreV1Api(self.api_client)
            self.apps_v1 = client.AppsV1Api(self.api_client)
            self.rbac_v1 = client.RbacAuthorizationV1Api(self.api_client)
            self.auth_v1 = client.AuthorizationV1Api(self.api_client)
            self.custom_objects = client.CustomObjectsApi(self.api_client)
            
            # Test the connection
            self._test_connection()
            
            self.logger.info("Successfully authenticated with Kubernetes API")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to authenticate with Kubernetes API: {e}")
            return False
    
    def authenticate_with_user_info(self, user_info: UserInfo, api_server_url: Optional[str] = None) -> bool:
        """
        Authenticate with the Kubernetes API using UserInfo from OAuth proxy.
        
        Args:
            user_info: UserInfo object containing bearer token and user details
            api_server_url: Optional API server URL override
            
        Returns:
            bool: True if authentication successful, False otherwise
        """
        if not user_info.bearer_token:
            self.logger.error("No bearer token found in UserInfo")
            return False
        
        try:
            # Store user information
            self._current_user = user_info
            self._store_session_info(user_info)
            
            # Use existing token authentication method
            success = self.authenticate_with_token(user_info.bearer_token, api_server_url)
            
            if success:
                self.logger.info(f"Successfully authenticated user: {user_info.username}")
            else:
                self._current_user = None
                self._clear_session_info()
                
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to authenticate with UserInfo: {e}")
            self._current_user = None
            self._clear_session_info()
            return False
    
    def _store_session_info(self, user_info: UserInfo) -> None:
        """
        Store user session information in Streamlit session state.
        
        Args:
            user_info: UserInfo object to store
        """
        try:
            if not hasattr(st, 'session_state'):
                self.logger.warning("Streamlit session state not available")
                return
                
            # Generate session identifier
            import hashlib
            session_data = f"{user_info.username}_{time.time()}"
            self._session_id = hashlib.md5(session_data.encode()).hexdigest()[:8]
            
            # Store in session state with proper cleanup
            st.session_state['k8s_client_session'] = {
                'session_id': self._session_id,
                'username': user_info.username,
                'preferred_username': user_info.preferred_username,
                'groups': user_info.groups,
                'authenticated_at': time.time(),
                'bearer_token': user_info.bearer_token  # Store for session continuity
            }
            
            self.logger.debug(f"Stored session info for user: {user_info.username}")
            
        except Exception as e:
            self.logger.error(f"Failed to store session info: {e}")
    
    def _clear_session_info(self) -> None:
        """Clear user session information from Streamlit session state."""
        try:
            if hasattr(st, 'session_state') and 'k8s_client_session' in st.session_state:
                del st.session_state['k8s_client_session']
                self.logger.debug("Cleared session info")
                
            self._session_id = None
            
        except Exception as e:
            self.logger.error(f"Failed to clear session info: {e}")
    
    def get_current_user(self) -> Optional[UserInfo]:
        """
        Get the currently authenticated user information.
        
        Returns:
            Optional[UserInfo]: Current user info or None if not authenticated
        """
        return self._current_user
    
    def is_authenticated(self) -> bool:
        """
        Check if the client is currently authenticated.
        
        Returns:
            bool: True if authenticated with valid session
        """
        return (
            self._current_user is not None and 
            self.api_client is not None and
            self._is_session_valid()
        )
    
    def _is_session_valid(self) -> bool:
        """
        Check if the current session is still valid.
        
        Returns:
            bool: True if session is valid, False otherwise
        """
        try:
            # For multi-threaded operations, we can't rely on Streamlit session state
            # Instead, check if we have the basic authentication components
            if not hasattr(st, 'session_state'):
                # In worker threads, session state is not available
                # Check if we have authenticated user and API client
                return (self._current_user is not None and 
                       self.api_client is not None and 
                       self._session_id is not None)
            
            # In main thread, check full session state
            if 'k8s_client_session' not in st.session_state:
                return False
                
            session = st.session_state['k8s_client_session']
            
            # Check session ID match
            if session.get('session_id') != self._session_id:
                return False
                
            # Check if session has required fields
            required_fields = ['username', 'bearer_token', 'authenticated_at']
            if not all(field in session for field in required_fields):
                return False
                
            # Session is valid if it exists and has required fields
            # OAuth proxy handles token expiration, so we don't check token validity here
            return True
            
        except Exception as e:
            self.logger.debug(f"Session validation check (thread-safe): {e}")
            # Fallback for thread safety - check basic authentication components
            return (self._current_user is not None and 
                   self.api_client is not None and 
                   self._session_id is not None)
    
    def refresh_session_from_state(self) -> bool:
        """
        Attempt to restore session from Streamlit session state.
        Useful for maintaining authentication across Streamlit reruns.
        
        Returns:
            bool: True if session was successfully restored
        """
        try:
            if not hasattr(st, 'session_state') or 'k8s_client_session' not in st.session_state:
                return False
                
            session = st.session_state['k8s_client_session']
            
            # Recreate UserInfo from session
            user_info = UserInfo(
                username=session['username'],
                preferred_username=session['preferred_username'],
                groups=session['groups'],
                bearer_token=session['bearer_token']
            )
            
            # Re-authenticate with stored token
            return self.authenticate_with_user_info(user_info)
            
        except Exception as e:
            self.logger.error(f"Failed to refresh session from state: {e}")
            return False
    
    def logout(self) -> None:
        """
        Clear authentication and session information.
        """
        try:
            self._current_user = None
            self._clear_session_info()
            
            # Close API client
            if self.api_client:
                self.api_client.close()
                
            # Reset all API instances
            self.api_client = None
            self.core_v1 = None
            self.apps_v1 = None
            self.rbac_v1 = None
            self.auth_v1 = None
            self.custom_objects = None
            
            # Clear cache
            self.clear_cache()
            
            self.logger.info("Successfully logged out and cleared session")
            
        except Exception as e:
            self.logger.error(f"Error during logout: {e}")
    
    def _ensure_authenticated(self) -> None:
        """
        Ensure the client is authenticated and session is valid.
        
        Raises:
            RuntimeError: If not authenticated or session is invalid
        """
        if not self.is_authenticated():
            if self._current_user is None:
                raise RuntimeError("Client not authenticated. Please authenticate with bearer token first.")
            elif self.api_client is None:
                raise RuntimeError("API client not initialized. Authentication may have failed.")
            else:
                raise RuntimeError("Session is invalid. Please re-authenticate.")
    
    def _format_api_error(self, api_exception: ApiException, operation: str) -> str:
        """
        Format API exception into user-friendly error message.
        
        Args:
            api_exception: The Kubernetes API exception
            operation: Description of the operation that failed
            
        Returns:
            str: Formatted error message
        """
        status_code = api_exception.status
        
        # Common status codes and their meanings
        error_messages = {
            401: "Authentication failed. Please check your bearer token.",
            403: f"Access denied. You don't have permission to {operation}.",
            404: f"Resource not found while trying to {operation}.",
            500: f"Kubernetes API server error while trying to {operation}.",
            503: f"Kubernetes API server unavailable while trying to {operation}."
        }
        
        base_message = error_messages.get(status_code, f"API error ({status_code}) while trying to {operation}")
        
        # Add specific error details if available
        try:
            if hasattr(api_exception, 'body') and api_exception.body:
                import json
                error_body = json.loads(api_exception.body)
                if 'message' in error_body:
                    base_message += f" Details: {error_body['message']}"
        except (json.JSONDecodeError, AttributeError):
            pass
            
        return base_message
    
    def _convert_api_exception(self, api_exception: ApiException, operation: str) -> K8sBaseException:
        """
        Convert ApiException to appropriate custom exception.
        
        Args:
            api_exception: The Kubernetes API exception
            operation: Description of the operation that failed
            
        Returns:
            K8sBaseException: Appropriate custom exception
        """
        status_code = api_exception.status
        context = {
            'operation': operation,
            'status_code': status_code,
            'reason': getattr(api_exception, 'reason', None),
            'body': getattr(api_exception, 'body', None)
        }
        
        # Add user context if available
        if self._current_user:
            context['user'] = self._current_user.username
            context['groups'] = self._current_user.groups
        
        error_message = self._format_api_error(api_exception, operation)
        
        if status_code == 401:
            return K8sAuthenticationError(error_message, status_code, operation, context)
        elif status_code == 403:
            return K8sAuthorizationError(error_message, status_code, operation, context)
        elif status_code == 404:
            return K8sNotFoundError(error_message, status_code, operation, context)
        elif status_code >= 500:
            return K8sServerError(error_message, status_code, operation, context)
        else:
            return K8sBaseException(error_message, status_code, operation, context)
    
    def _is_retryable_error(self, exception: Exception) -> bool:
        """
        Determine if an error is retryable.
        
        Args:
            exception: The exception to check
            
        Returns:
            bool: True if the error is retryable
        """
        # Network/timeout errors are retryable
        if isinstance(exception, (K8sNetworkError, K8sTimeoutError)):
            return True
        
        # Server errors (5xx) are usually retryable
        if isinstance(exception, K8sServerError):
            return True
        
        # Authentication and authorization errors are not retryable
        if isinstance(exception, (K8sAuthenticationError, K8sAuthorizationError)):
            return False
        
        # Not found errors are not retryable
        if isinstance(exception, K8sNotFoundError):
            return False
        
        # Circuit breaker errors are not retryable within this context
        if isinstance(exception, K8sCircuitBreakerError):
            return False
        
        # For ApiException, check specific status codes
        if isinstance(exception, ApiException):
            # Retryable status codes
            retryable_codes = {429, 500, 502, 503, 504}
            return exception.status in retryable_codes
        
        # For other exceptions, retry by default
        return True
    
    def _log_retry_attempt(self, operation: str, attempt: int, delay: float, exception: Exception) -> None:
        """
        Log retry attempt with detailed context.
        
        Args:
            operation: Name of the operation being retried
            attempt: Current attempt number
            delay: Delay before next attempt
            exception: Exception that triggered the retry
        """
        if not self.config.enable_detailed_logging:
            return
        
        context = {
            'operation': operation,
            'attempt': attempt,
            'delay': delay,
            'exception_type': type(exception).__name__,
            'exception_message': str(exception)
        }
        
        if self.config.log_request_context and self._current_user:
            context['user'] = self._current_user.username
            context['session_id'] = self._session_id
        
        self.logger.warning(
            f"Retry attempt {attempt} for {operation} in {delay:.2f}s due to {type(exception).__name__}: {exception}",
            extra={'context': context}
        )
    
    def _log_degradation(self, operation: str, reason: str) -> None:
        """
        Log graceful degradation event.
        
        Args:
            operation: Name of the operation being degraded
            reason: Reason for degradation
        """
        if not self.config.enable_detailed_logging:
            return
        
        context = {
            'operation': operation,
            'reason': reason,
            'degradation': True
        }
        
        if self.config.log_request_context and self._current_user:
            context['user'] = self._current_user.username
            context['session_id'] = self._session_id
        
        self.logger.info(
            f"Graceful degradation for {operation}: {reason}",
            extra={'context': context}
        )
    
    def _get_degraded_response(self, operation: str, *args, **kwargs) -> Any:
        """
        Get a degraded response when the operation fails.
        
        Args:
            operation: Name of the operation
            *args: Original operation arguments
            **kwargs: Original operation keyword arguments
            
        Returns:
            Appropriate degraded response based on operation type
        """
        # Map operations to appropriate degraded responses
        degraded_responses = {
            'list_namespaces': [],
            'discover_namespace_resources': {},
            'discover_custom_resource_definitions': [],
            'get_namespace': None,
            'discover_api_resources': {},
            'can_i': (False, "Service temporarily unavailable"),
            'can_i_batch': {},
            'discover_user_roles': [],
            'can_backup_resources': {},
            'can_restore_resources': {},
            '_discover_resources_in_namespace_fast': {},
            '_discover_crd_resources_in_namespace': {}
        }
        
        return degraded_responses.get(operation, None)
    
    def _create_connection_with_timeout(self, address, timeout=None, source_address=None, socket_options=None):
        """
        Create connection with enhanced timeout and pooling configuration.
        
        This method can be used to override the default connection behavior
        for better connection pooling and timeout handling.
        """
        import socket
        
        # Use the configuration timeouts
        connect_timeout = timeout or self.config.connection_timeout
        
        # Create socket with custom options
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        if source_address:
            sock.bind(source_address)
        
        if socket_options:
            for opt in socket_options:
                sock.setsockopt(*opt)
        
        # Set socket timeout
        sock.settimeout(connect_timeout)
        
        try:
            sock.connect(address)
            return sock
        except Exception as e:
            sock.close()
            raise K8sNetworkError(f"Failed to connect to {address}: {e}", context={'address': address, 'timeout': connect_timeout})
    
    def _test_connection(self) -> None:
        """
        Test the connection to the Kubernetes API.
        
        Raises:
            ApiException: If the connection test fails
        """
        try:
            # Simple API call to test connection
            if self.core_v1:
                self.core_v1.get_api_resources()
            self.logger.debug("Connection test successful")
        except ApiException as e:
            self.logger.error(f"Connection test failed: {e}")
            raise
    
    @with_retry_and_circuit_breaker("list_namespaces", critical=True)
    def list_namespaces(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        List all namespaces accessible to the authenticated user with automatic pagination.
        
        Args:
            use_cache: Whether to use cached results if available
        
        Returns:
            List[Dict[str, Any]]: List of namespace information
            
        Raises:
            RuntimeError: If client is not authenticated
            ApiException: If API call fails
        """
        self._ensure_authenticated()
        
        # Check cache first
        if use_cache and self._is_namespace_cache_valid():
            with self._cache_lock:
                self.logger.debug(f"Returning {len(self._namespaces_cache)} cached namespaces")
                return self._namespaces_cache.copy()
        
        try:
            result = []
            continue_token = None
            
            # Automatic pagination - fetch all namespaces
            while True:
                if continue_token:
                    namespaces = self.core_v1.list_namespace(_continue=continue_token, limit=100)
                else:
                    namespaces = self.core_v1.list_namespace(limit=100)
                
                # Process current batch
                for ns in namespaces.items:
                    result.append({
                        'name': ns.metadata.name,
                        'status': ns.status.phase,
                        'created': ns.metadata.creation_timestamp.isoformat() if ns.metadata.creation_timestamp else None,
                        'labels': ns.metadata.labels or {},
                        'annotations': ns.metadata.annotations or {},
                        'uid': ns.metadata.uid,
                        'resource_version': ns.metadata.resource_version
                    })
                
                # Check if there are more results
                continue_token = getattr(namespaces.metadata, 'continue', None)
                if not continue_token:
                    break
            
            # Update cache
            with self._cache_lock:
                self._namespaces_cache = result.copy()
                self._namespaces_cache_timestamp = time.time()
            
            self.logger.debug(f"Retrieved {len(result)} namespaces for user: {self._current_user.username if self._current_user else 'unknown'}")
            return result
            
        except ApiException as e:
            error_msg = self._format_api_error(e, "list namespaces")
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    @with_retry_and_circuit_breaker("discover_namespace_resources", critical=False)
    def discover_namespace_resources(self, namespace: str, use_cache: bool = True, 
                                   include_crds: Optional[bool] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Discover resources in a specific namespace, focusing on Velero-relevant types.
        
        Args:
            namespace: The namespace to discover resources in
            use_cache: Whether to use cached results if available
            include_crds: Override config setting for including CRD resources
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Resources grouped by type
            
        Raises:
            RuntimeError: If client is not authenticated or API call fails
        """
        self._ensure_authenticated()
        
        # Check cache first
        if use_cache and self._is_namespace_resource_cache_valid(namespace):
            with self._cache_lock:
                cached_resources = self._namespace_resources_cache.get(namespace, {})
                self.logger.debug(f"Returning cached resources for namespace '{namespace}': {len(cached_resources)} resource types")
                return cached_resources.copy()
        
        try:
            resources = {}
            
            # Determine if we should include CRDs
            should_include_crds = include_crds if include_crds is not None else self.config.include_crd_resources
            
            if self.config.fast_discovery:
                # Fast discovery: only check Velero-relevant resources
                resources = self._discover_velero_resources_fast(namespace)
            else:
                # Complete discovery: check all available resources
                resources = self._discover_all_namespace_resources(namespace)
            
            # Add CRD resources if enabled
            if should_include_crds:
                crd_resources = self._discover_crd_resources_in_namespace(namespace)
                resources.update(crd_resources)
            
            # Update cache
            with self._cache_lock:
                self._namespace_resources_cache[namespace] = resources.copy()
                self._namespace_cache_timestamps[namespace] = time.time()
            
            total_resources = sum(len(resource_list) for resource_list in resources.values())
            self.logger.debug(f"Discovered {total_resources} resources across {len(resources)} types in namespace '{namespace}'")
            
            return resources
            
        except Exception as e:
            error_msg = f"Failed to discover resources in namespace '{namespace}': {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    @with_retry_and_circuit_breaker("discover_custom_resource_definitions", critical=False)
    def discover_custom_resource_definitions(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Discover Custom Resource Definitions (CRDs) available in the cluster.
        
        Args:
            use_cache: Whether to use cached results if available
            
        Returns:
            List[Dict[str, Any]]: List of CRD information
            
        Raises:
            RuntimeError: If client is not authenticated or API call fails
        """
        self._ensure_authenticated()
        
        # Check cache first
        if use_cache and self._is_crd_cache_valid():
            with self._cache_lock:
                self.logger.debug(f"Returning {len(self._crd_cache)} cached CRDs")
                return self._crd_cache.copy()
        
        try:
            # Get API extensions client
            if not hasattr(self, 'extensions_v1'):
                self.extensions_v1 = client.ApiextensionsV1Api(self.api_client)
            
            result = []
            continue_token = None
            
            # Automatic pagination for CRDs
            while True:
                if continue_token:
                    crds = self.extensions_v1.list_custom_resource_definition(_continue=continue_token, limit=100)
                else:
                    crds = self.extensions_v1.list_custom_resource_definition(limit=100)
                
                # Process current batch
                for crd in crds.items:
                    # Extract basic metadata
                    crd_info = {
                        'name': crd.metadata.name,
                        'group': crd.spec.group,
                        'version': crd.spec.versions[0].name if crd.spec.versions else None,
                        'kind': crd.spec.names.kind,
                        'plural': crd.spec.names.plural,
                        'singular': crd.spec.names.singular,
                        'scope': crd.spec.scope,
                        'created': crd.metadata.creation_timestamp.isoformat() if crd.metadata.creation_timestamp else None,
                        'labels': crd.metadata.labels or {},
                        'api_version': f"{crd.spec.group}/{crd.spec.versions[0].name}" if crd.spec.versions else crd.spec.group
                    }
                    result.append(crd_info)
                
                # Check if there are more results
                continue_token = getattr(crds.metadata, 'continue', None)
                if not continue_token:
                    break
            
            # Update cache
            with self._cache_lock:
                self._crd_cache = result.copy()
                self._crd_cache_timestamp = time.time()
            
            self.logger.debug(f"Discovered {len(result)} Custom Resource Definitions")
            return result
            
        except ApiException as e:
            if e.status == 403:
                # User might not have permissions to list CRDs
                self.logger.warning("No permission to list Custom Resource Definitions")
                return []
            error_msg = self._format_api_error(e, "discover custom resource definitions")
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to discover Custom Resource Definitions: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    @with_retry_and_circuit_breaker("get_namespace", critical=False)
    def get_namespace(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific namespace.
        
        Args:
            name: Namespace name
            
        Returns:
            Optional[Dict[str, Any]]: Namespace information or None if not found
            
        Raises:
            RuntimeError: If client is not authenticated or API call fails
        """
        self._ensure_authenticated()
        
        try:
            ns = self.core_v1.read_namespace(name)
            return {
                'name': ns.metadata.name,
                'status': ns.status.phase,
                'created': ns.metadata.creation_timestamp.isoformat() if ns.metadata.creation_timestamp else None,
                'labels': ns.metadata.labels or {},
                'annotations': ns.metadata.annotations or {}
            }
            
        except ApiException as e:
            if e.status == 404:
                return None
            error_msg = self._format_api_error(e, f"get namespace '{name}'")
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def discover_api_resources(self, use_cache: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        Discover all available API resources in the cluster.
        
        Args:
            use_cache: Whether to use cached results if available
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: API resources grouped by API version
            
        Raises:
            RuntimeError: If client is not authenticated or API call fails
        """
        self._ensure_authenticated()
        
        # Check cache
        if use_cache and self._is_cache_valid():
            return self._api_resources_cache
        
        try:
            # Get API versions
            api_versions = self.api_client.call_api('/api', 'GET')[0]
            apis = self.api_client.call_api('/apis', 'GET')[0]
            
            resources = {}
            
            # Process core API resources
            for version in api_versions.get('versions', []):
                try:
                    version_resources = self.api_client.call_api(f'/api/{version}', 'GET')[0]
                    resources[f'api/{version}'] = version_resources.get('resources', [])
                except Exception as e:
                    self.logger.warning(f"Failed to get resources for api/{version}: {e}")
            
            # Process API group resources
            for group in apis.get('groups', []):
                group_name = group['name']
                for version in group.get('versions', []):
                    version_name = version['version']
                    api_version = f"{group_name}/{version_name}"
                    
                    try:
                        version_resources = self.api_client.call_api(f'/apis/{api_version}', 'GET')[0]
                        resources[f'apis/{api_version}'] = version_resources.get('resources', [])
                    except Exception as e:
                        self.logger.warning(f"Failed to get resources for apis/{api_version}: {e}")
            
            # Update cache
            self._api_resources_cache = resources
            self._cache_timestamp = time.time()
            
            self.logger.debug(f"Discovered {len(resources)} API versions")
            return resources
            
        except Exception as e:
            error_msg = f"Failed to discover API resources: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    @with_retry_and_circuit_breaker("can_i", critical=True)
    def can_i(self, verb: str, resource: str, namespace: Optional[str] = None, 
              use_cache: bool = True) -> Tuple[bool, Optional[str]]:
        """
        Check if the current user can perform a specific operation on a resource.
        
        Args:
            verb: The operation (e.g., 'get', 'list', 'create', 'delete')
            resource: The resource type (e.g., 'pods', 'namespaces')
            namespace: Optional namespace for namespaced resources
            use_cache: Whether to use cached results if available
            
        Returns:
            Tuple[bool, Optional[str]]: (allowed, error_message)
            - allowed: True if permission is granted
            - error_message: None if allowed, specific error message if denied
        """
        self._ensure_authenticated()
        
        # Create cache key
        cache_key = f"{verb}:{resource}:{namespace or '*'}"
        
        # Check cache first
        if use_cache and self._is_rbac_cache_valid():
            with self._cache_lock:
                if cache_key in self._rbac_cache:
                    allowed = self._rbac_cache[cache_key]
                    error_msg = None if allowed else f"Access denied: Cannot {verb} {resource}" + (f" in namespace '{namespace}'" if namespace else "")
                    self.logger.debug(f"RBAC cache hit for {cache_key}: {allowed}")
                    return allowed, error_msg
        
        try:
            # Create SelfSubjectAccessReview
            review = client.V1SelfSubjectAccessReview(
                spec=client.V1SelfSubjectAccessReviewSpec(
                    resource_attributes=client.V1ResourceAttributes(
                        resource=resource,
                        verb=verb,
                        namespace=namespace
                    )
                )
            )
            
            result = self.auth_v1.create_self_subject_access_review(body=review)
            allowed = result.status.allowed
            reason = getattr(result.status, 'reason', '') or ''
            
            # Update cache
            with self._cache_lock:
                self._rbac_cache[cache_key] = allowed
                self._rbac_cache_timestamp = time.time()
            
            # Prepare error message if denied
            error_msg = None
            if not allowed:
                context = f" in namespace '{namespace}'" if namespace else ""
                if reason:
                    error_msg = f"Access denied: Cannot {verb} {resource}{context}. Reason: {reason}"
                else:
                    error_msg = f"Access denied: Cannot {verb} {resource}{context}"
            
            self.logger.debug(f"RBAC check for {cache_key}: {allowed}")
            return allowed, error_msg
            
        except ApiException as e:
            error_msg = self._format_api_error(e, f"check RBAC permissions for {verb} {resource}")
            self.logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Failed to check permissions for {verb} {resource}: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    @with_retry_and_circuit_breaker("can_i_batch", critical=False)
    def can_i_batch(self, checks: List[Tuple[str, str, Optional[str]]], 
                   use_cache: bool = True) -> Dict[str, Tuple[bool, Optional[str]]]:
        """
        Check multiple permissions in parallel for better performance.
        
        Args:
            checks: List of (verb, resource, namespace) tuples to check
            use_cache: Whether to use cached results if available
            
        Returns:
            Dict[str, Tuple[bool, Optional[str]]]: Results keyed by "verb:resource:namespace"
            - Each value is (allowed, error_message)
        """
        self._ensure_authenticated()
        
        results = {}
        uncached_checks = []
        
        # Check cache first if enabled
        if use_cache and self._is_rbac_cache_valid():
            with self._cache_lock:
                for verb, resource, namespace in checks:
                    cache_key = f"{verb}:{resource}:{namespace or '*'}"
                    if cache_key in self._rbac_cache:
                        allowed = self._rbac_cache[cache_key]
                        error_msg = None if allowed else f"Access denied: Cannot {verb} {resource}" + (f" in namespace '{namespace}'" if namespace else "")
                        results[cache_key] = (allowed, error_msg)
                        self.logger.debug(f"RBAC cache hit for {cache_key}: {allowed}")
                    else:
                        uncached_checks.append((verb, resource, namespace))
        else:
            uncached_checks = checks
        
        # Process uncached checks in parallel
        if uncached_checks:
            def check_single_permission(check_tuple: Tuple[str, str, Optional[str]]) -> Tuple[str, bool, Optional[str]]:
                verb, resource, namespace = check_tuple
                cache_key = f"{verb}:{resource}:{namespace or '*'}"
                
                try:
                    # Create SelfSubjectAccessReview
                    review = client.V1SelfSubjectAccessReview(
                        spec=client.V1SelfSubjectAccessReviewSpec(
                            resource_attributes=client.V1ResourceAttributes(
                                resource=resource,
                                verb=verb,
                                namespace=namespace
                            )
                        )
                    )
                    
                    result = self.auth_v1.create_self_subject_access_review(body=review)
                    allowed = result.status.allowed
                    reason = getattr(result.status, 'reason', '') or ''
                    
                    # Prepare error message if denied
                    error_msg = None
                    if not allowed:
                        context = f" in namespace '{namespace}'" if namespace else ""
                        if reason:
                            error_msg = f"Access denied: Cannot {verb} {resource}{context}. Reason: {reason}"
                        else:
                            error_msg = f"Access denied: Cannot {verb} {resource}{context}"
                    
                    return cache_key, allowed, error_msg
                    
                except ApiException as e:
                    error_msg = self._format_api_error(e, f"check RBAC permissions for {verb} {resource}")
                    self.logger.error(error_msg)
                    return cache_key, False, error_msg
                except Exception as e:
                    error_msg = f"Failed to check permissions for {verb} {resource}: {e}"
                    self.logger.error(error_msg)
                    return cache_key, False, error_msg
            
            # Execute in parallel
            with ThreadPoolExecutor(max_workers=self.config.max_parallel_requests) as executor:
                future_to_check = {
                    executor.submit(check_single_permission, check_tuple): check_tuple
                    for check_tuple in uncached_checks
                }
                
                for future in as_completed(future_to_check):
                    try:
                        cache_key, allowed, error_msg = future.result()
                        results[cache_key] = (allowed, error_msg)
                        
                        # Update cache
                        with self._cache_lock:
                            self._rbac_cache[cache_key] = allowed
                            self._rbac_cache_timestamp = time.time()
                            
                    except Exception as e:
                        check_tuple = future_to_check[future]
                        verb, resource, namespace = check_tuple
                        cache_key = f"{verb}:{resource}:{namespace or '*'}"
                        error_msg = f"Failed to check permissions for {verb} {resource}: {e}"
                        results[cache_key] = (False, error_msg)
                        self.logger.error(error_msg)
        
        self.logger.debug(f"Batch RBAC check completed: {len(results)} permissions checked")
        return results
    
    @with_retry_and_circuit_breaker("discover_user_roles", critical=False)
    def discover_user_roles(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Discover roles and rolebindings that affect the current user.
        
        Args:
            use_cache: Whether to use cached results if available
            
        Returns:
            List[Dict[str, Any]]: List of role information affecting the current user
        """
        self._ensure_authenticated()
        
        # Check cache first
        if use_cache and self._is_user_roles_cache_valid():
            with self._cache_lock:
                self.logger.debug(f"Returning {len(self._user_roles_cache)} cached user roles")
                return self._user_roles_cache.copy()
        
        try:
            user_roles = []
            current_user = self._current_user
            
            if not current_user:
                return user_roles
            
            # Get accessible namespaces
            namespaces = self.list_namespaces(use_cache=use_cache)
            
            # Check roles in each accessible namespace
            for ns_info in namespaces:
                namespace = ns_info['name']
                
                try:
                    # List RoleBindings in this namespace
                    rolebindings = self.rbac_v1.list_namespaced_role_binding(namespace)
                    
                    for rb in rolebindings.items:
                        # Check if this rolebinding affects the current user
                        user_affected = False
                        
                        # Check subjects
                        for subject in rb.subjects or []:
                            if (subject.kind == 'User' and subject.name == current_user.username) or \
                               (subject.kind == 'Group' and subject.name in current_user.groups):
                                user_affected = True
                                break
                        
                        if user_affected:
                            # Get the role details
                            role_info = {
                                'type': 'Role',
                                'namespace': namespace,
                                'binding_name': rb.metadata.name,
                                'role_name': rb.role_ref.name,
                                'role_kind': rb.role_ref.kind,
                                'subjects': [],
                                'rules': []
                            }
                            
                            # Add subject information
                            for subject in rb.subjects or []:
                                role_info['subjects'].append({
                                    'kind': subject.kind,
                                    'name': subject.name,
                                    'namespace': getattr(subject, 'namespace', None)
                                })
                            
                            # Get role rules if it's a Role (not ClusterRole)
                            if rb.role_ref.kind == 'Role':
                                try:
                                    role = self.rbac_v1.read_namespaced_role(rb.role_ref.name, namespace)
                                    for rule in role.rules or []:
                                        role_info['rules'].append({
                                            'verbs': list(rule.verbs or []),
                                            'resources': list(rule.resources or []),
                                            'resource_names': list(rule.resource_names or []),
                                            'api_groups': list(rule.api_groups or [])
                                        })
                                except ApiException as e:
                                    if e.status != 404:  # Ignore not found, log others
                                        self.logger.debug(f"Could not read role {rb.role_ref.name}: {e}")
                            
                            user_roles.append(role_info)
                            
                except ApiException as e:
                    if e.status == 403:
                        # No permission to list rolebindings in this namespace
                        self.logger.debug(f"No permission to list rolebindings in namespace {namespace}")
                    else:
                        self.logger.warning(f"Error listing rolebindings in namespace {namespace}: {e}")
                except Exception as e:
                    self.logger.warning(f"Unexpected error listing rolebindings in namespace {namespace}: {e}")
            
            # Also check ClusterRoleBindings (these affect user globally)
            try:
                cluster_rolebindings = self.rbac_v1.list_cluster_role_binding()
                
                for crb in cluster_rolebindings.items:
                    # Check if this clusterrolebinding affects the current user
                    user_affected = False
                    
                    for subject in crb.subjects or []:
                        if (subject.kind == 'User' and subject.name == current_user.username) or \
                           (subject.kind == 'Group' and subject.name in current_user.groups):
                            user_affected = True
                            break
                    
                    if user_affected:
                        role_info = {
                            'type': 'ClusterRole',
                            'namespace': None,  # ClusterRoles are cluster-wide
                            'binding_name': crb.metadata.name,
                            'role_name': crb.role_ref.name,
                            'role_kind': crb.role_ref.kind,
                            'subjects': [],
                            'rules': []
                        }
                        
                        # Add subject information
                        for subject in crb.subjects or []:
                            role_info['subjects'].append({
                                'kind': subject.kind,
                                'name': subject.name,
                                'namespace': getattr(subject, 'namespace', None)
                            })
                        
                        # Note: We don't fetch ClusterRole rules here as they can be very extensive
                        # and we're focused on "what can I backup/restore"
                        
                        user_roles.append(role_info)
                        
            except ApiException as e:
                if e.status == 403:
                    self.logger.debug("No permission to list cluster role bindings")
                else:
                    self.logger.warning(f"Error listing cluster role bindings: {e}")
            except Exception as e:
                self.logger.warning(f"Unexpected error listing cluster role bindings: {e}")
            
            # Update cache
            with self._cache_lock:
                self._user_roles_cache = user_roles.copy()
                self._user_roles_cache_timestamp = time.time()
            
            self.logger.debug(f"Discovered {len(user_roles)} roles affecting current user")
            return user_roles
            
        except Exception as e:
            error_msg = f"Failed to discover user roles: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def can_backup_resources(self, resource_types: List[str], namespace: str) -> Dict[str, Tuple[bool, Optional[str]]]:
        """
        Check if the user can backup specific resource types in a namespace.
        
        Args:
            resource_types: List of resource types to check (e.g., ['pods', 'services'])
            namespace: Namespace to check permissions in
            
        Returns:
            Dict[str, Tuple[bool, Optional[str]]]: Results keyed by resource type
        """
        # For backup, we need 'list' and 'get' permissions
        checks = []
        for resource_type in resource_types:
            for verb in VELERO_OPERATIONS['backup']:
                checks.append((verb, resource_type, namespace))
        
        batch_results = self.can_i_batch(checks)
        
        # Aggregate results by resource type
        results = {}
        for resource_type in resource_types:
            resource_allowed = True
            error_messages = []
            
            for verb in VELERO_OPERATIONS['backup']:
                cache_key = f"{verb}:{resource_type}:{namespace}"
                allowed, error_msg = batch_results.get(cache_key, (False, f"Permission check failed for {verb} {resource_type}"))
                
                if not allowed:
                    resource_allowed = False
                    if error_msg:
                        error_messages.append(error_msg)
            
            final_error = "; ".join(error_messages) if error_messages else None
            results[resource_type] = (resource_allowed, final_error)
        
        return results
    
    def can_restore_resources(self, resource_types: List[str], namespace: str) -> Dict[str, Tuple[bool, Optional[str]]]:
        """
        Check if the user can restore specific resource types in a namespace.
        
        Args:
            resource_types: List of resource types to check (e.g., ['pods', 'services'])
            namespace: Namespace to check permissions in
            
        Returns:
            Dict[str, Tuple[bool, Optional[str]]]: Results keyed by resource type
        """
        # For restore, we need more permissions including 'create' and 'patch'
        checks = []
        for resource_type in resource_types:
            for verb in VELERO_OPERATIONS['restore']:
                checks.append((verb, resource_type, namespace))
        
        batch_results = self.can_i_batch(checks)
        
        # Aggregate results by resource type
        results = {}
        for resource_type in resource_types:
            resource_allowed = True
            error_messages = []
            
            for verb in VELERO_OPERATIONS['restore']:
                cache_key = f"{verb}:{resource_type}:{namespace}"
                allowed, error_msg = batch_results.get(cache_key, (False, f"Permission check failed for {verb} {resource_type}"))
                
                if not allowed:
                    resource_allowed = False
                    if error_msg:
                        error_messages.append(error_msg)
            
            final_error = "; ".join(error_messages) if error_messages else None
            results[resource_type] = (resource_allowed, final_error)
        
        return results
    
    def _is_cache_valid(self) -> bool:
        """Check if the current API resources cache is still valid."""
        return (
            self._api_resources_cache and
            time.time() - self._cache_timestamp < self.config.resource_cache_ttl
        )
    
    def clear_cache(self) -> None:
        """Clear all internal caches."""
        with self._cache_lock:
            self._api_resources_cache.clear()
            self._cache_timestamp = 0
            self._namespace_resources_cache.clear()
            self._namespace_cache_timestamps.clear()
            self._namespaces_cache.clear()
            self._namespaces_cache_timestamp = 0
            self._crd_cache.clear()
            self._crd_cache_timestamp = 0
            self._rbac_cache.clear()
            self._rbac_cache_timestamp = 0
            self._user_roles_cache.clear()
            self._user_roles_cache_timestamp = 0
        self.logger.debug("All caches cleared")
    
    def _is_namespace_cache_valid(self) -> bool:
        """Check if the namespace cache is still valid."""
        return (
            self._namespaces_cache and
            time.time() - self._namespaces_cache_timestamp < self.config.namespace_cache_ttl
        )
    
    def _is_namespace_resource_cache_valid(self, namespace: str) -> bool:
        """Check if the resource cache for a specific namespace is still valid."""
        return (
            namespace in self._namespace_resources_cache and
            namespace in self._namespace_cache_timestamps and
            time.time() - self._namespace_cache_timestamps[namespace] < self.config.resource_cache_ttl
        )
    
    def _is_crd_cache_valid(self) -> bool:
        """Check if the CRD cache is still valid."""
        return (
            self._crd_cache and
            time.time() - self._crd_cache_timestamp < self.config.resource_cache_ttl
        )
    
    def _is_rbac_cache_valid(self) -> bool:
        """Check if the RBAC permissions cache is still valid."""
        return (
            self._rbac_cache and
            time.time() - self._rbac_cache_timestamp < self.config.resource_cache_ttl
        )
    
    def _is_user_roles_cache_valid(self) -> bool:
        """Check if the user roles cache is still valid."""
        return (
            self._user_roles_cache and
            time.time() - self._user_roles_cache_timestamp < self.config.resource_cache_ttl
        )
    
    def _discover_velero_resources_fast(self, namespace: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fast discovery of Velero-relevant resources in a namespace using parallel requests.
        
        Args:
            namespace: The namespace to discover resources in
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Resources grouped by type
        """
        resources = {}
        
        # Prepare API clients for different resource types
        api_clients = {
            'v1': self.core_v1,
            'apps/v1': self.apps_v1,
            'batch/v1': client.BatchV1Api(self.api_client),
            'networking.k8s.io/v1': client.NetworkingV1Api(self.api_client),
            'rbac.authorization.k8s.io/v1': self.rbac_v1,
            'storage.k8s.io/v1': client.StorageV1Api(self.api_client)
        }
        
        def fetch_resource_type(api_version: str, resource_type: str) -> Tuple[str, List[Dict[str, Any]]]:
            """Fetch a specific resource type from the namespace."""
            try:
                api_client = api_clients.get(api_version)
                if not api_client:
                    return f"{api_version}/{resource_type}", []
                
                # Get the appropriate list method
                list_method_name = f"list_namespaced_{resource_type.rstrip('s')}"
                if hasattr(api_client, list_method_name):
                    list_method = getattr(api_client, list_method_name)
                    
                    # Fetch resources with pagination
                    result_items = []
                    continue_token = None
                    
                    while True:
                        if continue_token:
                            response = list_method(namespace, _continue=continue_token, limit=100)
                        else:
                            response = list_method(namespace, limit=100)
                        
                        # Process current batch
                        for item in response.items:
                            result_items.append({
                                'name': item.metadata.name,
                                'namespace': item.metadata.namespace,
                                'created': item.metadata.creation_timestamp.isoformat() if item.metadata.creation_timestamp else None,
                                'labels': item.metadata.labels or {},
                                'annotations': item.metadata.annotations or {},
                                'uid': item.metadata.uid,
                                'resource_version': item.metadata.resource_version
                            })
                        
                        # Check for more results
                        continue_token = getattr(response.metadata, 'continue', None)
                        if not continue_token:
                            break
                    
                    return f"{api_version}/{resource_type}", result_items
                else:
                    return f"{api_version}/{resource_type}", []
                    
            except ApiException as e:
                if e.status == 403:
                    # User doesn't have permission for this resource type
                    self.logger.debug(f"No permission to list {api_version}/{resource_type} in namespace {namespace}")
                    return f"{api_version}/{resource_type}", []
                elif e.status == 404:
                    # Resource type not available in this cluster
                    self.logger.debug(f"Resource type {api_version}/{resource_type} not available in cluster")
                    return f"{api_version}/{resource_type}", []
                else:
                    self.logger.warning(f"Error fetching {api_version}/{resource_type} in namespace {namespace}: {e}")
                    return f"{api_version}/{resource_type}", []
            except Exception as e:
                self.logger.warning(f"Unexpected error fetching {api_version}/{resource_type} in namespace {namespace}: {e}")
                return f"{api_version}/{resource_type}", []
        
        # Collect all tasks for parallel execution
        tasks = []
        for api_version, resource_types in VELERO_RELEVANT_RESOURCES.items():
            for resource_type in resource_types:
                tasks.append((api_version, resource_type))
        
        # Execute requests in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_requests) as executor:
            future_to_resource = {
                executor.submit(fetch_resource_type, api_version, resource_type): (api_version, resource_type)
                for api_version, resource_type in tasks
            }
            
            for future in as_completed(future_to_resource):
                try:
                    resource_key, resource_items = future.result()
                    if resource_items:  # Only include resource types that have items
                        resources[resource_key] = resource_items
                except Exception as e:
                    api_version, resource_type = future_to_resource[future]
                    self.logger.error(f"Failed to fetch {api_version}/{resource_type}: {e}")
        
        return resources
    
    def _discover_all_namespace_resources(self, namespace: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Complete discovery of all available resources in a namespace (slower but comprehensive).
        
        Args:
            namespace: The namespace to discover resources in
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Resources grouped by type
        """
        # This method would use the existing discover_api_resources and then
        # fetch all discovered resource types. For now, we'll use the fast method
        # as the base and can expand this later if needed.
        return self._discover_velero_resources_fast(namespace)
    
    def _discover_crd_resources_in_namespace(self, namespace: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Discover custom resources (CRD instances) in a namespace.
        
        Args:
            namespace: The namespace to discover custom resources in
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Custom resources grouped by type
        """
        try:
            # First get all CRDs
            crds = self.discover_custom_resource_definitions()
            
            custom_resources = {}
            
            # For each namespaced CRD, try to list instances in the namespace
            for crd in crds:
                if crd['scope'] == 'Namespaced':
                    try:
                        # Use custom objects API to list instances
                        instances = self.custom_objects.list_namespaced_custom_object(
                            group=crd['group'],
                            version=crd['version'],
                            namespace=namespace,
                            plural=crd['plural']
                        )
                        
                        resource_items = []
                        for item in instances.get('items', []):
                            metadata = item.get('metadata', {})
                            resource_items.append({
                                'name': metadata.get('name'),
                                'namespace': metadata.get('namespace'),
                                'created': metadata.get('creationTimestamp'),
                                'labels': metadata.get('labels', {}),
                                'annotations': metadata.get('annotations', {}),
                                'uid': metadata.get('uid'),
                                'resource_version': metadata.get('resourceVersion')
                            })
                        
                        if resource_items:
                            custom_resources[crd['api_version'] + '/' + crd['plural']] = resource_items
                            
                    except ApiException as e:
                        if e.status not in [403, 404]:  # Ignore permission and not found errors
                            self.logger.debug(f"Could not list custom resource {crd['plural']} in namespace {namespace}: {e}")
                    except Exception as e:
                        self.logger.debug(f"Error listing custom resource {crd['plural']} in namespace {namespace}: {e}")
            
            return custom_resources
            
        except Exception as e:
            self.logger.warning(f"Failed to discover custom resources in namespace {namespace}: {e}")
            return {}
    
    def close(self) -> None:
        """Close the client and clean up resources."""
        if self.api_client:
            self.api_client.close()
        self.logger.info("KubernetesClient closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit."""
        self.close()


def get_k8s_client() -> KubernetesClient:
    """
    Get or create a singleton KubernetesClient instance for Streamlit session.
    
    This function manages a singleton instance of KubernetesClient in Streamlit's
    session state, ensuring proper session continuity across reruns.
    
    Returns:
        KubernetesClient: Configured client instance
    """
    if not hasattr(st, 'session_state'):
        # Fallback for non-Streamlit environments
        return KubernetesClient()
    
    # Check if client exists in session state
    if 'k8s_client_instance' not in st.session_state:
        st.session_state['k8s_client_instance'] = KubernetesClient()
    
    client_instance = st.session_state['k8s_client_instance']
    
    # Try to restore session if client is not authenticated
    if not client_instance.is_authenticated():
        client_instance.refresh_session_from_state()
    
    return client_instance