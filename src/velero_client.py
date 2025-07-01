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
    command_timeout: int = 300  # 5 minutes for backup/restore operations
    
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
        
        try:
            # Create temporary kubeconfig for authentication
            kubeconfig_path = self._create_kubeconfig_file()
            
            # Set up environment with kubeconfig
            env = os.environ.copy()
            env["KUBECONFIG"] = kubeconfig_path
            
            self.logger.info(f"Executing Velero command: {' '.join(cmd_args)}")
            
            # Execute command
            result = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                timeout=self.config.command_timeout,
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
            error_msg = f"Velero command timed out after {self.config.command_timeout} seconds"
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
        """
        self.logger.info("Listing all Velero backups")
        
        # Build get backups command with JSON output
        command = VeleroCommand(self.config.binary_path, self.config.velero_namespace)
        command.get_backups(output_format=output_format)
        
        try:
            result = self._execute_command(command)
            
            if not result.success:
                self.logger.error(f"Failed to list backups: {result.stderr}")
                return []
            
            # Parse JSON output if requested
            if output_format.lower() == "json":
                import json
                try:
                    if result.stdout.strip():
                        data = json.loads(result.stdout)
                        # Velero JSON output structure: {"items": [...]}
                        if isinstance(data, dict) and "items" in data:
                            return data["items"]
                        elif isinstance(data, list):
                            return data
                        else:
                            self.logger.warning(f"Unexpected JSON structure in backup list: {type(data)}")
                            return []
                    else:
                        # Empty output means no backups
                        return []
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse JSON output from velero: {e}")
                    return []
            else:
                # For non-JSON formats, return raw output as single item
                return [{"raw_output": result.stdout}] if result.stdout.strip() else []
                
        except (VeleroCommandError, VeleroCircuitBreakerError) as e:
            self.logger.error(f"Error listing backups: {e}")
            # Return empty list instead of raising - let UI handle the error display
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error listing backups: {e}")
            return []
    
    def get_backup_status(self, backup_name: str) -> str:
        """
        Get the status of a specific Velero backup.
        
        Args:
            backup_name: Name of the backup to check
            
        Returns:
            Status string (e.g., "Completed", "InProgress", "Failed", "NotFound")
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
                        return phase
                    elif isinstance(status, str):
                        return status
                    else:
                        return "Unknown"
            
            # Backup not found
            return "NotFound"
            
        except Exception as e:
            self.logger.error(f"Error getting backup status for '{backup_name}': {e}")
            return "Error"
    
    def get_current_user(self) -> Optional[UserInfo]:
        """Get current authenticated user from Kubernetes client."""
        return self.k8s_client.get_current_user()
    
    def is_authenticated(self) -> bool:
        """Check if client is properly authenticated."""
        return self.k8s_client.is_authenticated()


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