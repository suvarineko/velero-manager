"""
Velero Binary Integration Module.

This module provides a client for interacting with the Velero binary to perform
backup and restore operations in Kubernetes clusters.
"""

import logging
import os
import subprocess
import tempfile
import time
import random
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from functools import wraps
from enum import Enum

# Try relative imports first, fallback for standalone testing
try:
    from .k8s_client import KubernetesClient
    from .auth.auth import UserInfo
except ImportError:
    from k8s_client import KubernetesClient
    from auth.auth import UserInfo


# Circuit Breaker State Enum
class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


# Exception Hierarchy
class VeleroBaseException(Exception):
    """Base exception for all Velero-related errors."""
    def __init__(self, message: str, exit_code: Optional[int] = None, 
                 operation: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.exit_code = exit_code
        self.operation = operation
        self.context = context or {}
        self.timestamp = time.time()


class VeleroConfigurationError(VeleroBaseException):
    """Raised when Velero client configuration is invalid."""
    pass


class VeleroBinaryError(VeleroBaseException):
    """Raised when Velero binary execution fails."""
    pass


class VeleroCommandError(VeleroBaseException):
    """Raised when Velero command execution returns non-zero exit code."""
    
    def __init__(self, message: str, exit_code: int, stderr: str = ""):
        super().__init__(message, exit_code=exit_code)
        self.stderr = stderr


class VeleroAuthenticationError(VeleroBaseException):
    """Raised when authentication with Kubernetes fails for Velero operations."""
    pass


class VeleroCircuitBreakerError(VeleroBaseException):
    """Raised when circuit breaker is open."""
    pass


class VeleroRetryableError(VeleroBaseException):
    """Raised for errors that should trigger retry logic."""
    pass


# Granular Exception Classes for Specific Velero Scenarios
class VeleroError(VeleroBaseException):
    """General Velero operation error."""
    pass


class VeleroParsingError(VeleroBaseException):
    """Raised when Velero output cannot be parsed."""
    pass


class BackupNotFoundError(VeleroBaseException):
    """Raised when a requested backup does not exist."""
    pass


class BackupInProgressError(VeleroBaseException):
    """Raised when trying to operate on a backup that is still in progress."""
    pass


class BackupFailedError(VeleroBaseException):
    """Raised when a backup operation has failed."""
    pass


class BackupCompletedError(VeleroBaseException):
    """Raised when trying to modify a completed backup."""
    pass


class RestoreNotFoundError(VeleroBaseException):
    """Raised when a requested restore does not exist."""
    pass


class RestoreInProgressError(VeleroBaseException):
    """Raised when trying to operate on a restore that is still in progress."""
    pass


class RestoreFailedError(VeleroBaseException):
    """Raised when a restore operation has failed."""
    pass


class RestorePartialFailureError(VeleroBaseException):
    """Raised when a restore completes with partial failures."""
    pass


class RestoreCompletedError(VeleroBaseException):
    """Raised when trying to modify a completed restore."""
    pass


@dataclass
class CircuitBreaker:
    """
    Circuit breaker implementation for Velero operations.
    
    Protects against cascading failures by temporarily stopping
    requests when failure rate exceeds threshold.
    """
    failure_threshold: int = 5
    timeout: float = 60.0  # seconds
    half_open_max_calls: int = 3
    
    def __post_init__(self):
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
        self._lock = threading.RLock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time >= self.timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    raise VeleroCircuitBreakerError("Circuit breaker is open")
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    raise VeleroCircuitBreakerError("Circuit breaker half-open limit reached")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure(e)
                raise
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_calls += 1
    
    def _on_failure(self, exception: Exception):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


def with_retry_and_circuit_breaker(operation_name: str = None):
    """
    Decorator that adds retry logic and circuit breaker protection to Velero methods.
    
    Args:
        operation_name: Name of the operation for logging
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
            
            def execute_with_retry():
                last_exception = None
                
                for attempt in range(config.max_retries + 1):
                    try:
                        if circuit_breaker:
                            return circuit_breaker.call(func, self, *args, **kwargs)
                        else:
                            return func(self, *args, **kwargs)
                    
                    except VeleroCircuitBreakerError:
                        # Circuit breaker is open, fail fast
                        self.logger.error(f"Circuit breaker open for operation '{op_name}'")
                        raise
                    
                    except Exception as e:
                        last_exception = e
                        
                        # Check if this is a retryable error
                        if not self._is_retryable_error(e):
                            self.logger.error(f"Non-retryable error in '{op_name}': {e}")
                            raise
                        
                        # Don't retry on the last attempt
                        if attempt == config.max_retries:
                            break
                        
                        # Calculate delay with exponential backoff and jitter
                        delay = min(
                            config.base_retry_delay * (2 ** attempt),
                            config.max_retry_delay
                        )
                        
                        if config.retry_jitter:
                            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
                        
                        self.logger.warning(
                            f"Attempt {attempt + 1}/{config.max_retries + 1} failed for '{op_name}': {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                
                # All retries exhausted
                self.logger.error(f"All retry attempts exhausted for '{op_name}': {last_exception}")
                raise last_exception
            
            return execute_with_retry()
        return wrapper
    return decorator


@dataclass
class VeleroClientConfig:
    """
    Configuration for Velero client operations.
    
    Configuration focusing on essential parameters plus retry/circuit breaker settings.
    Values are typically sourced from Helm chart values.yaml.
    """
    # Core Velero configuration
    velero_namespace: str = "velero"
    binary_path: str = "/usr/local/bin/velero"
    default_backup_ttl: str = "720h"
    backup_storage_location: str = "default"
    
    # Timeout configuration - two categories for different operation types
    quick_operation_timeout: int = 60    # 1 minute for list, get, status operations
    long_operation_timeout: int = 1800   # 30 minutes for backup, restore operations
    command_timeout: int = 300           # Fallback timeout (5 minutes)
    
    # Retry configuration (same defaults as k8s_client)
    max_retries: int = 3
    base_retry_delay: float = 1.0  # Base delay for exponential backoff
    max_retry_delay: float = 60.0  # Maximum delay between retries
    retry_jitter: bool = True  # Add random jitter to prevent thundering herd
    
    # Circuit breaker configuration (same defaults as k8s_client)
    enable_circuit_breaker: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    circuit_breaker_half_open_max_calls: int = 3


@dataclass
class VeleroCommandResult:
    """Result of a Velero command execution."""
    exit_code: int
    stdout: str
    stderr: str
    command: List[str]
    
    @property
    def success(self) -> bool:
        """Returns True if command executed successfully."""
        return self.exit_code == 0


class VeleroCommand:
    """
    Command builder for Velero binary operations.
    
    Provides a fluent interface for building Velero commands before execution.
    """
    
    def __init__(self, binary_path: str, namespace: str):
        self.binary_path = binary_path
        self.namespace = namespace
        self.args = [binary_path]
        self.logger = logging.getLogger(__name__)
    
    def backup(self, name: str, **kwargs) -> 'VeleroCommand':
        """
        Add backup create command with specified name.
        
        Args:
            name: Name of the backup
            **kwargs: Additional backup options (ttl, include_namespaces, etc.)
        """
        self.args.extend(["backup", "create", name])
        self.args.extend(["--namespace", self.namespace])
        
        # Handle common backup options
        if "ttl" in kwargs:
            self.args.extend(["--ttl", kwargs["ttl"]])
        if "include_namespaces" in kwargs:
            namespaces = kwargs["include_namespaces"]
            if isinstance(namespaces, list):
                namespaces = ",".join(namespaces)
            self.args.extend(["--include-namespaces", namespaces])
        if "storage_location" in kwargs:
            self.args.extend(["--storage-location", kwargs["storage_location"]])
        
        return self
    
    def restore(self, backup_name: str, **kwargs) -> 'VeleroCommand':
        """
        Add restore create command from specified backup.
        
        Args:
            backup_name: Name of the backup to restore from
            **kwargs: Additional restore options (restore_name, include_namespaces, etc.)
        """
        restore_name = kwargs.get("restore_name", f"{backup_name}-restore")
        self.args.extend(["restore", "create", restore_name])
        self.args.extend(["--from-backup", backup_name])
        self.args.extend(["--namespace", self.namespace])
        
        # Handle common restore options
        if "include_namespaces" in kwargs:
            namespaces = kwargs["include_namespaces"]
            if isinstance(namespaces, list):
                namespaces = ",".join(namespaces)
            self.args.extend(["--include-namespaces", namespaces])
        if "namespace_mappings" in kwargs:
            for old_ns, new_ns in kwargs["namespace_mappings"].items():
                self.args.extend(["--namespace-mappings", f"{old_ns}:{new_ns}"])
        
        return self
    
    def get_backups(self, **kwargs) -> 'VeleroCommand':
        """
        Add get backups command.
        
        Args:
            **kwargs: Additional options (output_format, selector, etc.)
        """
        self.args.extend(["backup", "get"])
        self.args.extend(["--namespace", self.namespace])
        
        if "output_format" in kwargs:
            self.args.extend(["-o", kwargs["output_format"]])
        if "selector" in kwargs:
            self.args.extend(["--selector", kwargs["selector"]])
        
        return self
    
    def get_restores(self, **kwargs) -> 'VeleroCommand':
        """
        Add get restores command.
        
        Args:
            **kwargs: Additional options (output_format, selector, etc.)
        """
        self.args.extend(["restore", "get"])
        self.args.extend(["--namespace", self.namespace])
        
        if "output_format" in kwargs:
            self.args.extend(["-o", kwargs["output_format"]])
        if "selector" in kwargs:
            self.args.extend(["--selector", kwargs["selector"]])
        
        return self
    
    def build(self) -> List[str]:
        """Build and return the final command arguments."""
        return self.args.copy()
    
    def get_operation_type(self) -> str:
        """
        Determine the operation type from command arguments.
        
        Returns:
            'long' for long-running operations, 'quick' for quick operations
        """
        if len(self.args) < 2:
            return 'quick'  # Default for simple commands
        
        # Check for long-running operations
        long_operations = {
            ('backup', 'create'),
            ('restore', 'create'),
        }
        
        # Check for quick operations  
        quick_operations = {
            ('backup', 'get'),
            ('backup', 'describe'),
            ('restore', 'get'),
            ('restore', 'describe'),
            ('version',),
            ('get',),
        }
        
        # Extract command parts (skip binary path)
        command_parts = tuple(self.args[1:3]) if len(self.args) >= 3 else tuple(self.args[1:])
        
        if command_parts in long_operations:
            return 'long'
        elif command_parts in quick_operations or command_parts[:1] in quick_operations:
            return 'quick'
        else:
            return 'quick'  # Default to quick for unknown operations


class VeleroClient:
    """
    Client for Velero binary operations with Kubernetes integration.
    
    Provides high-level interface for backup and restore operations
    using the Velero CLI binary with tight Kubernetes client integration.
    """
    
    def __init__(self, k8s_client: KubernetesClient, config: Optional[VeleroClientConfig] = None):
        """
        Initialize Velero client with Kubernetes integration.
        
        Args:
            k8s_client: Authenticated KubernetesClient instance
            config: Optional configuration object (uses defaults if not provided)
        """
        if not k8s_client or not k8s_client.is_authenticated():
            raise VeleroAuthenticationError("VeleroClient requires authenticated KubernetesClient")
        
        self.k8s_client = k8s_client
        self.config = config or VeleroClientConfig()
        self.logger = logging.getLogger(__name__)
        
        # Load configuration from environment (Helm values)
        self._load_config_from_env()
        
        # Validate configuration
        self._validate_configuration()
        
        # Initialize circuit breaker if enabled
        if self.config.enable_circuit_breaker:
            self._circuit_breaker = CircuitBreaker(
                failure_threshold=self.config.circuit_breaker_failure_threshold,
                timeout=self.config.circuit_breaker_timeout,
                half_open_max_calls=self.config.circuit_breaker_half_open_max_calls
            )
        else:
            self._circuit_breaker = None
        
        self.logger.info(f"VeleroClient initialized - namespace: {self.config.velero_namespace}, "
                        f"binary: {self.config.binary_path}, "
                        f"circuit_breaker: {self.config.enable_circuit_breaker}")
    
    def _load_config_from_env(self) -> None:
        """Load configuration from environment variables (Helm values)."""
        if os.getenv("VELERO_NAMESPACE"):
            self.config.velero_namespace = os.getenv("VELERO_NAMESPACE")
        if os.getenv("VELERO_DEFAULT_BACKUP_TTL"):
            self.config.default_backup_ttl = os.getenv("VELERO_DEFAULT_BACKUP_TTL")
        if os.getenv("BACKUP_STORAGE_LOCATION"):
            self.config.backup_storage_location = os.getenv("BACKUP_STORAGE_LOCATION")
        if os.getenv("VELERO_BINARY_PATH"):
            self.config.binary_path = os.getenv("VELERO_BINARY_PATH")
    
    def _validate_configuration(self) -> None:
        """Validate Velero client configuration."""
        # Check if Velero binary exists
        if not Path(self.config.binary_path).exists():
            raise VeleroConfigurationError(f"Velero binary not found at {self.config.binary_path}")
        
        # Check if binary is executable
        if not os.access(self.config.binary_path, os.X_OK):
            raise VeleroConfigurationError(f"Velero binary at {self.config.binary_path} is not executable")
        
        # Validate namespace format
        if not self.config.velero_namespace or not self.config.velero_namespace.strip():
            raise VeleroConfigurationError("Velero namespace cannot be empty")
        
        # Validate timeout configurations
        if self.config.quick_operation_timeout <= 0:
            raise VeleroConfigurationError("Quick operation timeout must be positive")
        if self.config.long_operation_timeout <= 0:
            raise VeleroConfigurationError("Long operation timeout must be positive")
        if self.config.command_timeout <= 0:
            raise VeleroConfigurationError("Command timeout must be positive")
        
        # Validate timeout ranges (quick should be less than long)
        if self.config.quick_operation_timeout >= self.config.long_operation_timeout:
            raise VeleroConfigurationError(
                f"Quick operation timeout ({self.config.quick_operation_timeout}s) should be less than "
                f"long operation timeout ({self.config.long_operation_timeout}s)"
            )
        
        # Validate reasonable timeout limits
        if self.config.quick_operation_timeout > 300:  # 5 minutes
            self.logger.warning(f"Quick operation timeout ({self.config.quick_operation_timeout}s) seems high")
        if self.config.long_operation_timeout > 7200:  # 2 hours
            self.logger.warning(f"Long operation timeout ({self.config.long_operation_timeout}s) seems very high")
        
        # Validate namespace name format (basic Kubernetes naming rules)
        import re
        if not re.match(r'^[a-z0-9]([a-z0-9\-]*[a-z0-9])?$', self.config.velero_namespace):
            raise VeleroConfigurationError(
                f"Invalid namespace format: '{self.config.velero_namespace}'. "
                "Must be lowercase alphanumeric with hyphens, starting and ending with alphanumeric."
            )
        
        # Validate TTL format (basic check)
        if not re.match(r'^\d+[hms]$', self.config.default_backup_ttl):
            raise VeleroConfigurationError(
                f"Invalid backup TTL format: '{self.config.default_backup_ttl}'. "
                "Must be a number followed by 'h', 'm', or 's' (e.g., '720h', '30m', '3600s')"
            )
        
        self.logger.debug("Velero client configuration validated successfully")
    
    def _is_retryable_error(self, exception: Exception) -> bool:
        """
        Determine if an error should trigger retry logic.
        
        Args:
            exception: The exception to check
            
        Returns:
            bool: True if the error is retryable, False otherwise
        """
        # Non-retryable errors (fail fast)
        if isinstance(exception, (VeleroConfigurationError, VeleroAuthenticationError)):
            return False
        
        # Command errors with specific exit codes that shouldn't be retried
        if isinstance(exception, VeleroCommandError):
            # Common non-retryable exit codes for Velero
            non_retryable_codes = {
                1,   # General CLI usage errors
                126, # Permission denied
                127, # Command not found
            }
            if exception.exit_code in non_retryable_codes:
                return False
            
            # Check stderr for specific non-retryable error patterns
            if exception.stderr:
                stderr_lower = exception.stderr.lower()
                non_retryable_patterns = [
                    "not found",
                    "already exists", 
                    "invalid",
                    "forbidden",
                    "unauthorized",
                    "permission denied"
                ]
                if any(pattern in stderr_lower for pattern in non_retryable_patterns):
                    return False
        
        # Retryable errors
        retryable_types = (
            subprocess.TimeoutExpired,    # Command timeouts
            VeleroBinaryError,           # Binary execution issues
            VeleroRetryableError,        # Explicitly marked as retryable
            ConnectionError,             # Network issues
            OSError,                     # System-level errors
        )
        
        if isinstance(exception, retryable_types):
            return True
        
        # VeleroCommandError with retryable exit codes
        if isinstance(exception, VeleroCommandError):
            # Temporary failure exit codes
            retryable_codes = {
                2,   # Misuse of shell builtins (could be temporary)
                130, # Script terminated by Control-C
                143, # Terminated by SIGTERM
            }
            return exception.exit_code in retryable_codes
        
        # Default to not retryable for unknown errors
        return False
    
    def _create_kubeconfig_file(self) -> str:
        """
        Create temporary kubeconfig file with current user's token.
        
        Returns:
            Path to temporary kubeconfig file
        """
        user_info = self.k8s_client.get_current_user()
        if not user_info or not user_info.bearer_token:
            raise VeleroAuthenticationError("No valid user token available for Velero operations")
        
        # Create temporary kubeconfig
        kubeconfig_content = f"""
apiVersion: v1
kind: Config
clusters:
- cluster:
    server: https://kubernetes.default.svc
    certificate-authority: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
  name: default-cluster
contexts:
- context:
    cluster: default-cluster
    user: {user_info.username}
  name: default-context
current-context: default-context
users:
- name: {user_info.username}
  user:
    token: {user_info.bearer_token}
"""
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.kubeconfig', delete=False) as f:
            f.write(kubeconfig_content.strip())
            return f.name
    
    @with_retry_and_circuit_breaker("execute_command")
    def _execute_command(self, command: VeleroCommand) -> VeleroCommandResult:
        """
        Execute Velero command with proper authentication and error handling.
        
        Args:
            command: VeleroCommand instance to execute
            
        Returns:
            VeleroCommandResult with execution details
        """
        cmd_args = command.build()
        kubeconfig_path = None
        
        # Determine appropriate timeout based on operation type
        operation_type = command.get_operation_type()
        if operation_type == 'long':
            timeout = self.config.long_operation_timeout
            self.logger.debug(f"Using long operation timeout: {timeout}s")
        elif operation_type == 'quick':
            timeout = self.config.quick_operation_timeout
            self.logger.debug(f"Using quick operation timeout: {timeout}s")
        else:
            timeout = self.config.command_timeout
            self.logger.debug(f"Using fallback timeout: {timeout}s")
        
        try:
            # Create temporary kubeconfig for authentication
            kubeconfig_path = self._create_kubeconfig_file()
            
            # Set up environment with kubeconfig
            env = os.environ.copy()
            env["KUBECONFIG"] = kubeconfig_path
            
            self.logger.info(f"Executing Velero command ({operation_type} operation): {' '.join(cmd_args)}")
            
            # Execute command with operation-specific timeout
            result = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )
            
            command_result = VeleroCommandResult(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                command=cmd_args
            )
            
            if not command_result.success:
                error_msg = f"Velero command failed with exit code {result.returncode}"
                self.logger.error(f"{error_msg}: {result.stderr}")
                raise VeleroCommandError(error_msg, result.returncode, result.stderr)
            
            self.logger.info("Velero command executed successfully")
            return command_result
            
        except subprocess.TimeoutExpired:
            error_msg = f"Velero {operation_type} operation timed out after {timeout} seconds"
            self.logger.error(error_msg)
            raise VeleroBinaryError(error_msg)
        except FileNotFoundError:
            error_msg = f"Velero binary not found at {self.config.binary_path}"
            self.logger.error(error_msg)
            raise VeleroBinaryError(error_msg)
        finally:
            # Clean up temporary kubeconfig
            if kubeconfig_path and os.path.exists(kubeconfig_path):
                os.unlink(kubeconfig_path)
    
    def _parse_output(self, result: VeleroCommandResult, expected_format: str = "json") -> Dict[str, Any]:
        """
        Parse Velero command output into structured Python objects.
        
        Args:
            result: VeleroCommandResult from command execution
            expected_format: Expected output format (currently only "json" supported)
            
        Returns:
            Parsed output as dictionary
            
        Raises:
            VeleroParsingError: If output cannot be parsed
            VeleroCommandError: If command failed
        """
        # Check if command was successful
        if not result.success:
            self.logger.error(f"Velero command failed (exit code {result.exit_code}): {result.stderr}")
            raise VeleroCommandError(
                f"Velero command failed: {result.stderr or 'Unknown error'}", 
                exit_code=result.exit_code,
                stderr=result.stderr
            )
        
        # Validate that we have output
        if not result.stdout or not result.stdout.strip():
            self.logger.warning("Velero command returned empty output")
            return {"items": []}
        
        # Currently only supporting JSON format
        if expected_format.lower() != "json":
            raise VeleroParsingError(f"Unsupported output format: {expected_format}")
        
        try:
            import json
            data = json.loads(result.stdout.strip())
            
            # Basic validation - ensure we have valid JSON structure
            if not isinstance(data, (dict, list)):
                raise VeleroParsingError(f"Invalid JSON structure: expected dict or list, got {type(data)}")
            
            # Normalize Velero output format
            if isinstance(data, dict):
                # Velero typically returns {"items": [...]} or {"kind": "...", "metadata": {...}}
                if "items" in data:
                    # List response (e.g., backup list, restore list)
                    if not isinstance(data["items"], list):
                        raise VeleroParsingError("Invalid JSON structure: 'items' field must be a list")
                    return data
                elif "kind" in data and "metadata" in data:
                    # Single resource response
                    return {"items": [data]}
                else:
                    # Unexpected structure, but valid JSON
                    self.logger.warning(f"Unexpected JSON structure: {list(data.keys())}")
                    return {"items": [data]}
            elif isinstance(data, list):
                # Direct list response
                return {"items": data}
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON output: {e}")
            self.logger.debug(f"Raw output was: {result.stdout[:500]}...")
            raise VeleroParsingError(f"Invalid JSON output from Velero: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error parsing Velero output: {e}")
            raise VeleroParsingError(f"Failed to parse Velero output: {e}")
    
    def _validate_backup_object(self, backup: Dict[str, Any]) -> None:
        """
        Validate a backup object has required fields.
        
        Args:
            backup: Backup object dictionary
            
        Raises:
            VeleroParsingError: If backup object is invalid
        """
        if not isinstance(backup, dict):
            raise VeleroParsingError("Backup object must be a dictionary")
        
        # Check for required metadata
        if "metadata" not in backup:
            raise VeleroParsingError("Backup object missing 'metadata' field")
        
        metadata = backup["metadata"]
        if not isinstance(metadata, dict):
            raise VeleroParsingError("Backup metadata must be a dictionary")
        
        if "name" not in metadata:
            raise VeleroParsingError("Backup metadata missing 'name' field")
    
    def _validate_restore_object(self, restore: Dict[str, Any]) -> None:
        """
        Validate a restore object has required fields.
        
        Args:
            restore: Restore object dictionary
            
        Raises:
            VeleroParsingError: If restore object is invalid
        """
        if not isinstance(restore, dict):
            raise VeleroParsingError("Restore object must be a dictionary")
        
        # Check for required metadata
        if "metadata" not in restore:
            raise VeleroParsingError("Restore object missing 'metadata' field")
        
        metadata = restore["metadata"]
        if not isinstance(metadata, dict):
            raise VeleroParsingError("Restore metadata must be a dictionary")
        
        if "name" not in metadata:
            raise VeleroParsingError("Restore metadata missing 'name' field")
    
    def create_backup(self, name: str, include_namespaces: Optional[List[str]] = None, 
                     ttl: Optional[str] = None, **kwargs) -> VeleroCommandResult:
        """
        Create a Velero backup.
        
        Args:
            name: Name of the backup
            include_namespaces: List of namespaces to include in backup
            ttl: Time to live for the backup (defaults to config value)
            **kwargs: Additional backup options
            
        Returns:
            VeleroCommandResult with backup operation details
        """
        self.logger.info(f"Creating backup: {name}")
        
        # Use default TTL if not specified
        backup_ttl = ttl or self.config.default_backup_ttl
        
        # Build backup command
        command = VeleroCommand(self.config.binary_path, self.config.velero_namespace)
        command.backup(
            name=name,
            ttl=backup_ttl,
            include_namespaces=include_namespaces,
            storage_location=self.config.backup_storage_location,
            **kwargs
        )
        
        return self._execute_command(command)
    
    def create_restore(self, backup_name: str, restore_name: Optional[str] = None,
                      include_namespaces: Optional[List[str]] = None,
                      namespace_mappings: Optional[Dict[str, str]] = None,
                      **kwargs) -> VeleroCommandResult:
        """
        Create a Velero restore from backup.
        
        Args:
            backup_name: Name of the backup to restore from
            restore_name: Name for the restore operation (auto-generated if not provided)
            include_namespaces: List of namespaces to include in restore
            namespace_mappings: Dictionary mapping old namespace names to new ones
            **kwargs: Additional restore options
            
        Returns:
            VeleroCommandResult with restore operation details
        """
        self.logger.info(f"Creating restore from backup: {backup_name}")
        
        # Build restore command
        command = VeleroCommand(self.config.binary_path, self.config.velero_namespace)
        command.restore(
            backup_name=backup_name,
            restore_name=restore_name,
            include_namespaces=include_namespaces,
            namespace_mappings=namespace_mappings,
            **kwargs
        )
        
        return self._execute_command(command)
    
    def list_backups(self, output_format: str = "json") -> List[Dict[str, Any]]:
        """
        List all Velero backups and return structured data.
        
        Args:
            output_format: Output format for velero command (default: "json")
            
        Returns:
            List of backup information dictionaries
            
        Raises:
            VeleroCommandError: If velero command fails after retries
            VeleroCircuitBreakerError: If circuit breaker is open
            VeleroParsingError: If output parsing fails
        """
        self.logger.info("Listing all Velero backups")
        
        # Build get backups command with JSON output
        command = VeleroCommand(self.config.binary_path, self.config.velero_namespace)
        command.get_backups(output_format=output_format)
        
        try:
            result = self._execute_command(command)
            parsed_output = self._parse_output(result, expected_format=output_format)
            
            # Validate each backup object
            backups = parsed_output.get("items", [])
            for backup in backups:
                self._validate_backup_object(backup)
            
            return backups
                
        except (VeleroCommandError, VeleroCircuitBreakerError, VeleroParsingError) as e:
            self.logger.error(f"Error listing backups: {e}")
            # Re-raise specific exceptions for UI to handle appropriately
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error listing backups: {e}")
            raise VeleroError(f"Failed to list backups: {e}")
    
    def get_backup_status(self, backup_name: str, raise_on_error: bool = False) -> str:
        """
        Get the status of a specific Velero backup.
        
        Args:
            backup_name: Name of the backup to check
            raise_on_error: If True, raise specific exceptions; if False, return error strings
            
        Returns:
            Status string (e.g., "Completed", "InProgress", "Failed", "NotFound", "Error")
            
        Raises:
            When raise_on_error=True:
                BackupNotFoundError: If backup doesn't exist
                BackupInProgressError: If backup is currently in progress
                BackupFailedError: If backup has failed
                VeleroError: For other backup-related errors
        """
        self.logger.info(f"Getting status for backup: {backup_name}")
        
        try:
            # Get all backups and find the specific one
            backups = self.list_backups()
            
            for backup in backups:
                # Check both 'name' and 'metadata.name' fields for backup name
                backup_id = backup.get("metadata", {}).get("name") or backup.get("name")
                
                if backup_id == backup_name:
                    # Extract status from backup object
                    status = backup.get("status", {})
                    if isinstance(status, dict):
                        phase = status.get("phase", "Unknown")
                        
                        # Handle based on raise_on_error flag
                        if raise_on_error:
                            if phase.lower() == "inprogress":
                                raise BackupInProgressError(f"Backup '{backup_name}' is currently in progress")
                            elif phase.lower() in ["failed", "partiallyfailed"]:
                                error_msg = status.get("errors", "Unknown error")
                                raise BackupFailedError(f"Backup '{backup_name}' failed: {error_msg}")
                        
                        return phase
                    elif isinstance(status, str):
                        return status
                    else:
                        return "Unknown"
            
            # Backup not found
            if raise_on_error:
                raise BackupNotFoundError(f"Backup '{backup_name}' not found")
            else:
                return "NotFound"
            
        except (BackupNotFoundError, BackupInProgressError, BackupFailedError):
            # Re-raise specific backup exceptions only if raise_on_error is True
            if raise_on_error:
                raise
            else:
                return "Error"
        except (VeleroCommandError, VeleroCircuitBreakerError, VeleroParsingError) as e:
            self.logger.error(f"Error getting backup status for '{backup_name}': {e}")
            if raise_on_error:
                raise VeleroError(f"Failed to get backup status: {e}")
            else:
                return "Error"
        except Exception as e:
            self.logger.error(f"Unexpected error getting backup status for '{backup_name}': {e}")
            if raise_on_error:
                raise VeleroError(f"Unexpected error getting backup status: {e}")
            else:
                return "Error"
    
    def get_current_user(self) -> Optional[UserInfo]:
        """Get current authenticated user from Kubernetes client."""
        return self.k8s_client.get_current_user()
    
    def is_authenticated(self) -> bool:
        """Check if client is properly authenticated."""
        return self.k8s_client.is_authenticated()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a basic health check of the Velero installation.
        
        Returns:
            Dictionary with health check results
        """
        health_status = {
            "overall_status": "unknown",
            "checks": {
                "binary_accessible": False,
                "version_command": False,
                "cluster_connectivity": False,
                "authentication": False
            },
            "details": {},
            "errors": []
        }
        
        try:
            # Check 1: Binary accessibility (already covered in validation, but let's be explicit)
            try:
                if Path(self.config.binary_path).exists() and os.access(self.config.binary_path, os.X_OK):
                    health_status["checks"]["binary_accessible"] = True
                    health_status["details"]["binary_path"] = self.config.binary_path
                else:
                    health_status["errors"].append(f"Velero binary not accessible at {self.config.binary_path}")
            except Exception as e:
                health_status["errors"].append(f"Binary check failed: {e}")
            
            # Check 2: Velero version command
            try:
                version_command = VeleroCommand(self.config.binary_path, self.config.velero_namespace)
                version_command.args.extend(["version", "--client-only"])
                
                # Execute version command (this should be quick)
                result = subprocess.run(
                    version_command.build(),
                    capture_output=True,
                    text=True,
                    timeout=self.config.quick_operation_timeout
                )
                
                if result.returncode == 0:
                    health_status["checks"]["version_command"] = True
                    # Extract version info if available
                    if "Version:" in result.stdout:
                        health_status["details"]["client_version"] = result.stdout.strip()
                else:
                    health_status["errors"].append(f"Version command failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                health_status["errors"].append("Version command timed out")
            except Exception as e:
                health_status["errors"].append(f"Version command error: {e}")
            
            # Check 3: Authentication status
            try:
                if self.is_authenticated():
                    health_status["checks"]["authentication"] = True
                    user_info = self.get_current_user()
                    if user_info:
                        health_status["details"]["authenticated_user"] = user_info.username
                else:
                    health_status["errors"].append("Not authenticated with Kubernetes cluster")
            except Exception as e:
                health_status["errors"].append(f"Authentication check failed: {e}")
            
            # Check 4: Basic cluster connectivity (try to get server version through Velero)
            try:
                if health_status["checks"]["authentication"]:
                    # Try a simple velero command that requires cluster access
                    server_version_command = VeleroCommand(self.config.binary_path, self.config.velero_namespace)
                    server_version_command.args.extend(["version"])
                    
                    result = self._execute_command(server_version_command)
                    if result.success:
                        health_status["checks"]["cluster_connectivity"] = True
                        if "Server:" in result.stdout:
                            health_status["details"]["server_version"] = result.stdout.strip()
                    else:
                        health_status["errors"].append("Failed to connect to Velero server in cluster")
                else:
                    health_status["errors"].append("Skipping cluster connectivity check (not authenticated)")
                    
            except Exception as e:
                health_status["errors"].append(f"Cluster connectivity check failed: {e}")
            
            # Determine overall status
            passed_checks = sum(health_status["checks"].values())
            total_checks = len(health_status["checks"])
            
            if passed_checks == total_checks:
                health_status["overall_status"] = "healthy"
            elif passed_checks >= total_checks / 2:
                health_status["overall_status"] = "degraded"
            else:
                health_status["overall_status"] = "unhealthy"
            
            health_status["details"]["passed_checks"] = f"{passed_checks}/{total_checks}"
            
            self.logger.info(f"Health check completed: {health_status['overall_status']} ({passed_checks}/{total_checks} checks passed)")
            
            return health_status
            
        except Exception as e:
            health_status["overall_status"] = "error"
            health_status["errors"].append(f"Health check failed: {e}")
            self.logger.error(f"Health check encountered error: {e}")
            return health_status


def get_velero_client(k8s_client: KubernetesClient, 
                     config: Optional[VeleroClientConfig] = None) -> VeleroClient:
    """
    Factory function to create VeleroClient instance.
    
    Args:
        k8s_client: Authenticated KubernetesClient instance
        config: Optional configuration (uses defaults if not provided)
        
    Returns:
        Configured VeleroClient instance
    """
    return VeleroClient(k8s_client=k8s_client, config=config)