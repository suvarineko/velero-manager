"""
Namespace Manager module for Velero Manager application.

This module provides namespace discovery with RBAC filtering capabilities,
leveraging multithreaded operations for efficient namespace management.
It includes admin permission checking, caching, and sorting functionality.
"""

import logging
import time
import random
import math
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from enum import Enum

# Import Kubernetes client integration
try:
    from .k8s_client import KubernetesClient, K8sClientConfig
except ImportError:
    # Fallback for standalone testing
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from k8s_client import KubernetesClient, K8sClientConfig


class SortOrder(Enum):
    """Enum for namespace sorting options"""
    NAME_ASC = "name_asc"
    NAME_DESC = "name_desc"
    CREATED_ASC = "created_asc"
    CREATED_DESC = "created_desc"
    ADMIN_ACCESS_FIRST = "admin_access_first"


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, calls blocked
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Simple circuit breaker for API operations"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    
    def __post_init__(self):
        self.failure_count: int = 0
        self.last_failure_time: float = 0.0
        self.state: CircuitBreakerState = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if not self._can_execute():
            raise Exception("Circuit breaker is OPEN - operation blocked")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _can_execute(self) -> bool:
        """Check if operation can be executed"""
        with self._lock:
            if self.state == CircuitBreakerState.CLOSED:
                return True
            elif self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    return True
                return False
            else:  # HALF_OPEN
                return True
    
    def _on_success(self):
        """Handle successful operation"""
        with self._lock:
            self.failure_count = 0
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        """Handle failed operation"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN


class RetryManager:
    """Exponential backoff retry manager"""
    
    def __init__(self, max_retries: int = 3, initial_delay: float = 1.0, 
                 max_delay: float = 30.0, backoff_multiplier: float = 2.0):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
    
    def execute_with_retry(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    # Final attempt failed
                    break
                
                # Calculate delay with jitter
                delay = min(
                    self.initial_delay * (self.backoff_multiplier ** attempt),
                    self.max_delay
                )
                # Add jitter (±20%)
                jitter = delay * 0.2 * (random.random() * 2 - 1)
                delay_with_jitter = max(0.1, delay + jitter)
                
                time.sleep(delay_with_jitter)
        
        # All retries failed
        raise last_exception


@dataclass
class NamespaceInfo:
    """Data class for namespace information with admin access details"""
    name: str
    status: str
    created: Optional[str] = None
    labels: Optional[Dict[str, str]] = None
    annotations: Optional[Dict[str, str]] = None
    has_admin_access: bool = False
    admin_permissions: Optional[Dict[str, bool]] = None
    resource_version: Optional[str] = None
    uid: Optional[str] = None
    last_checked: Optional[float] = None


@dataclass
class CacheEntry:
    """Cache entry with TTL support"""
    data: Any
    timestamp: float
    ttl: float
    
    def is_valid(self) -> bool:
        """Check if cache entry is still valid"""
        return time.time() - self.timestamp < self.ttl


@dataclass
class NamespaceManagerConfig:
    """Configuration for NamespaceManager"""
    # Caching configuration
    cache_ttl: float = 300.0  # 5 minutes default
    rbac_cache_ttl: float = 600.0  # 10 minutes for RBAC results
    enable_cache: bool = True
    
    # Threading configuration
    max_concurrent_workers: int = 10
    rbac_check_timeout: float = 30.0
    
    # Admin permission definitions  
    admin_verbs: Set[str] = field(default_factory=lambda: {
        'create', 'delete', 'deletecollection', 'patch', 'update'
    })
    admin_resources: Set[str] = field(default_factory=lambda: {
        'pods', 'deployments', 'secrets', 'configmaps', 'services',
        'persistentvolumeclaims', 'roles', 'rolebindings'
    })
    
    # Admin access requires ALL permissions to pass (no threshold)
    
    # Performance tracking configuration
    enable_performance_tracking: bool = True
    track_cache_hit_miss: bool = True
    
    # Error handling and resilience configuration
    enable_circuit_breaker: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: float = 60.0
    enable_retry_logic: bool = True
    max_retries: int = 3
    initial_retry_delay: float = 1.0
    max_retry_delay: float = 30.0
    retry_backoff_multiplier: float = 2.0
    
    # Memory optimization configuration
    large_cluster_threshold: int = 500  # Consider cluster "large" if > 500 namespaces
    batch_size_large_clusters: int = 50  # Process in batches for large clusters
    enable_memory_optimization: bool = True
    
    # Graceful degradation configuration
    allow_partial_rbac_failures: bool = True
    fallback_to_cached_data: bool = True
    
    # Filtering and sorting
    default_sort_order: SortOrder = SortOrder.NAME_ASC
    include_system_namespaces: bool = False
    system_namespace_prefixes: Set[str] = field(default_factory=lambda: {
        'kube-', 'velero', 'cattle-', 'istio-', 'cert-manager'
    })


class NamespaceManager:
    """
    Manages namespace discovery with RBAC filtering and caching.
    
    This class provides multithreaded namespace discovery that efficiently
    determines user admin permissions across namespaces using the Kubernetes
    SelfSubjectAccessReview API. It includes comprehensive caching, sorting,
    and filtering capabilities.
    """
    
    def __init__(self, 
                 k8s_client: KubernetesClient,
                 config: Optional[NamespaceManagerConfig] = None):
        """
        Initialize the NamespaceManager.
        
        Args:
            k8s_client: Configured KubernetesClient instance
            config: Configuration object for the manager
        """
        self.k8s_client = k8s_client
        self.config = config or NamespaceManagerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Separate caching layer for NamespaceManager
        self._namespace_cache: Dict[str, CacheEntry] = {}
        self._rbac_cache: Dict[str, CacheEntry] = {}
        self._discovery_cache: Optional[CacheEntry] = None
        
        # Thread safety for caching operations
        self._cache_lock = threading.RLock()
        
        # Performance tracking
        self._performance_stats = {
            'namespace_cache_hits': 0,
            'namespace_cache_misses': 0,
            'rbac_cache_hits': 0,
            'rbac_cache_misses': 0,
            'discovery_cache_hits': 0,
            'discovery_cache_misses': 0,
            'total_discovery_calls': 0,
            'total_rbac_calls': 0,
            'average_discovery_time': 0.0,
            'average_rbac_time': 0.0,
            'circuit_breaker_trips': 0,
            'retry_attempts': 0,
            'graceful_degradations': 0
        }
        
        # Error handling and resilience components
        if self.config.enable_circuit_breaker:
            self._discovery_circuit_breaker = CircuitBreaker(
                failure_threshold=self.config.circuit_breaker_failure_threshold,
                recovery_timeout=self.config.circuit_breaker_recovery_timeout
            )
            self._rbac_circuit_breaker = CircuitBreaker(
                failure_threshold=self.config.circuit_breaker_failure_threshold,
                recovery_timeout=self.config.circuit_breaker_recovery_timeout
            )
        else:
            self._discovery_circuit_breaker = None
            self._rbac_circuit_breaker = None
        
        if self.config.enable_retry_logic:
            self._retry_manager = RetryManager(
                max_retries=self.config.max_retries,
                initial_delay=self.config.initial_retry_delay,
                max_delay=self.config.max_retry_delay,
                backoff_multiplier=self.config.retry_backoff_multiplier
            )
        else:
            self._retry_manager = None
        
        self.logger.info("NamespaceManager initialized with config: %s", self.config)
    
    def discover_namespaces(self, 
                           force_refresh: bool = False,
                           include_rbac_check: bool = True,
                           sort_order: Optional[SortOrder] = None) -> List[NamespaceInfo]:
        """
        Discover namespaces with optional RBAC filtering using multithreading.
        
        Args:
            force_refresh: Force cache refresh
            include_rbac_check: Include admin permission checking
            sort_order: Sorting order for results
            
        Returns:
            List[NamespaceInfo]: List of namespace information with admin access details
        """
        start_time = time.time()
        self.logger.info("Starting namespace discovery - force_refresh=%s, include_rbac_check=%s", 
                        force_refresh, include_rbac_check)
        
        # Track discovery operation
        discovery_start_time = time.time()
        
        # Step 1: Check cache first (unless force refresh)
        if not force_refresh and self.config.enable_cache:
            cached_result = self.get_cached_namespaces()
            if cached_result is not None:
                self.logger.info("Returning cached namespace discovery results (%d namespaces)", 
                               len(cached_result))
                # Apply sorting if requested and different from cached
                if sort_order is not None:
                    cached_result = self.sort_namespaces(cached_result, sort_order)
                return cached_result
        
        try:
            # Step 2: Fetch raw namespace data from Kubernetes API with resilience
            def fetch_namespaces():
                return self.k8s_client.list_namespaces(use_cache=not force_refresh)
            
            raw_namespaces = self._execute_with_resilience(
                "discovery", 
                fetch_namespaces,
                self._discovery_circuit_breaker
            )
            
            if not raw_namespaces:
                self.logger.warning("No namespaces found from Kubernetes API")
                return []
            
            self.logger.info("Fetched %d raw namespaces from API", len(raw_namespaces))
            
            # Step 3: Process namespaces with memory-conscious batching
            namespace_infos = self._process_namespaces_in_batches(
                raw_namespaces, 
                include_rbac_check
            )
            
            # Step 4: Apply sorting
            final_order = sort_order or self.config.default_sort_order
            namespace_infos = self.sort_namespaces(namespace_infos, final_order)
            
            # Step 5: Cache the results
            if self.config.enable_cache:
                self._cache_discovery_results(namespace_infos)
            
            duration = time.time() - start_time
            discovery_duration = time.time() - discovery_start_time
            
            # Track performance metrics
            self._track_operation_time('discovery', discovery_duration)
            
            self.logger.info("Namespace discovery completed in %.2fs - found %d namespaces", 
                           duration, len(namespace_infos))
            
            return namespace_infos
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error("Namespace discovery failed after %.2fs: %s", duration, e)
            raise
    
    def check_rbac_access(self, 
                         namespace: str,
                         use_cache: bool = True) -> Tuple[bool, Dict[str, bool]]:
        """
        Check admin permissions for a specific namespace using SelfSubjectAccessReview.
        
        Uses the Kubernetes SelfSubjectAccessReview API to check all configured admin
        permissions. Returns admin access only if ALL permission checks pass.
        
        Args:
            namespace: Namespace to check permissions for
            use_cache: Whether to use cached results
            
        Returns:
            Tuple[bool, Dict[str, bool]]: (has_admin_access, detailed_permissions)
        """
        start_time = time.time()
        self.logger.debug("Checking RBAC access for namespace '%s' (use_cache=%s)", 
                         namespace, use_cache)
        
        # Step 1: Check cache first
        if use_cache and self.config.enable_cache:
            cached_result = self._get_cached_rbac_result(namespace)
            if cached_result is not None:
                has_admin, permissions = cached_result
                self.logger.debug("RBAC cache hit for namespace '%s': admin=%s", 
                                namespace, has_admin)
                return has_admin, permissions
        
        try:
            # Step 2: Generate all permission checks for admin access
            permission_checks = self._generate_admin_permission_checks(namespace)
            
            self.logger.debug("Checking %d admin permissions for namespace '%s'", 
                            len(permission_checks), namespace)
            
            # Step 3: Use k8s_client.can_i_batch() for parallel SelfSubjectAccessReview calls with resilience
            def check_permissions():
                return self.k8s_client.can_i_batch(permission_checks, use_cache=use_cache)
            
            batch_results = self._execute_with_resilience(
                "rbac",
                check_permissions,
                self._rbac_circuit_breaker
            )
            
            # Step 4: Process results - ALL permissions must pass for admin access
            detailed_permissions = {}
            all_passed = True
            
            for verb, resource, ns in permission_checks:
                cache_key = f"{verb}:{resource}:{ns or '*'}"
                
                if cache_key in batch_results:
                    allowed, error_msg = batch_results[cache_key]
                    permission_key = f"{verb}:{resource}"
                    detailed_permissions[permission_key] = allowed
                    
                    if not allowed:
                        all_passed = False
                        self.logger.debug("Permission denied for %s in namespace '%s': %s", 
                                        permission_key, namespace, error_msg or "Access denied")
                else:
                    # If a permission check is missing from results, consider it denied
                    permission_key = f"{verb}:{resource}"
                    detailed_permissions[permission_key] = False
                    all_passed = False
                    self.logger.warning("Missing permission result for %s in namespace '%s'", 
                                      permission_key, namespace)
            
            # Step 5: Determine admin access (ALL permissions must pass)
            has_admin_access = all_passed and len(detailed_permissions) > 0
            
            # Step 6: Cache the result
            if self.config.enable_cache:
                self._cache_rbac_result(namespace, has_admin_access, detailed_permissions)
            
            duration = time.time() - start_time
            granted_count = sum(1 for allowed in detailed_permissions.values() if allowed)
            total_count = len(detailed_permissions)
            
            # Track performance metrics
            self._track_operation_time('rbac', duration)
            
            self.logger.info(
                "RBAC check completed for namespace '%s' in %.2fs: admin=%s (%d/%d permissions granted)",
                namespace, duration, has_admin_access, granted_count, total_count
            )
            
            return has_admin_access, detailed_permissions
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error("RBAC check failed for namespace '%s' after %.2fs: %s", 
                            namespace, duration, e)
            
            # Try graceful degradation with cached data
            if self.config.allow_partial_rbac_failures and use_cache:
                cached_result = self._get_cached_rbac_result(namespace)
                if cached_result is not None:
                    has_admin, permissions = cached_result
                    self.logger.info("Using cached RBAC data for namespace '%s' due to API failure", namespace)
                    if self.config.enable_performance_tracking:
                        self._performance_stats['graceful_degradations'] += 1
                    return has_admin, permissions
            
            # Return safe defaults on error
            return False, {}
    
    def get_cached_namespaces(self) -> Optional[List[NamespaceInfo]]:
        """
        Get cached namespace discovery results if valid.
        
        Returns:
            Optional[List[NamespaceInfo]]: Cached results or None if invalid/missing
        """
        with self._cache_lock:
            if not self.config.enable_cache or not self._discovery_cache:
                self._track_cache_miss('discovery')
                return None
            
            if self._discovery_cache.is_valid():
                self._track_cache_hit('discovery')
                self.logger.debug("Returning cached namespace discovery results")
                return self._discovery_cache.data
            else:
                self._track_cache_miss('discovery')
                # Remove expired cache entry
                self._discovery_cache = None
            
            return None
    
    def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """
        Clear cached data.
        
        Args:
            cache_type: Type of cache to clear ('namespace', 'rbac', 'all', or None for all)
        """
        # Method stub - will be implemented in subsequent tasks
        with self._cache_lock:
            if cache_type in (None, 'all', 'namespace'):
                self._namespace_cache.clear()
                self._discovery_cache = None
                self.logger.debug("Cleared namespace cache")
            
            if cache_type in (None, 'all', 'rbac'):
                self._rbac_cache.clear()
                self.logger.debug("Cleared RBAC cache")
    
    def sort_namespaces(self, 
                       namespaces: List[NamespaceInfo],
                       sort_order: Optional[SortOrder] = None) -> List[NamespaceInfo]:
        """
        Sort namespaces according to specified criteria.
        
        Args:
            namespaces: List of namespaces to sort
            sort_order: Sorting order to apply
            
        Returns:
            List[NamespaceInfo]: Sorted list of namespaces
        """
        if not namespaces:
            return namespaces
            
        order = sort_order or self.config.default_sort_order
        self.logger.debug("Sorting %d namespaces with order=%s", len(namespaces), order.value)
        
        try:
            if order == SortOrder.NAME_ASC:
                # Case-insensitive ascending sort by name with None-safe handling
                return sorted(namespaces, key=lambda ns: self._get_sort_key_name(ns.name))
            elif order == SortOrder.NAME_DESC:
                # Case-insensitive descending sort by name with None-safe handling
                return sorted(namespaces, key=lambda ns: self._get_sort_key_name(ns.name), reverse=True)
            elif order == SortOrder.CREATED_ASC:
                # Ascending sort by creation time, None values last
                return sorted(namespaces, key=lambda ns: self._get_sort_key_created(ns.created))
            elif order == SortOrder.CREATED_DESC:
                # Descending sort by creation time, None values last
                return sorted(namespaces, key=lambda ns: self._get_sort_key_created(ns.created), reverse=True)
            elif order == SortOrder.ADMIN_ACCESS_FIRST:
                # Admin access first, then by name (stable secondary sort)
                return sorted(namespaces, key=lambda ns: (not ns.has_admin_access, self._get_sort_key_name(ns.name)))
            else:
                self.logger.warning("Unknown sort order: %s, using default NAME_ASC", order)
                return sorted(namespaces, key=lambda ns: self._get_sort_key_name(ns.name))
                
        except Exception as e:
            self.logger.error("Error sorting namespaces: %s", e)
            # Return original list on error
            return namespaces
    
    def _get_sort_key_name(self, name: Optional[str]) -> str:
        """
        Get a case-insensitive, None-safe sort key for namespace names.
        
        Args:
            name: Namespace name (can be None)
            
        Returns:
            str: Sort key (empty string for None, lowercase for others)
        """
        if name is None:
            return ""
        return name.lower()
    
    def _get_sort_key_created(self, created: Optional[str]) -> tuple:
        """
        Get a sort key for creation timestamps with None-safe handling.
        
        Args:
            created: ISO 8601 timestamp string (can be None)
            
        Returns:
            tuple: (is_none, parsed_datetime) for proper sorting
        """
        if created is None:
            # None values sort last (is_none=True sorts after is_none=False)
            return (True, "")
        
        try:
            # Parse ISO 8601 timestamp
            from datetime import datetime
            parsed_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
            return (False, parsed_dt)
        except (ValueError, AttributeError) as e:
            self.logger.debug("Failed to parse creation timestamp '%s': %s", created, e)
            # Invalid dates sort last but before None
            return (True, created or "")
    
    def filter_namespaces(self, 
                         namespaces: List[NamespaceInfo],
                         include_system: Optional[bool] = None,
                         admin_only: bool = False,
                         name_pattern: Optional[str] = None) -> List[NamespaceInfo]:
        """
        Filter namespaces based on various criteria.
        
        Args:
            namespaces: List of namespaces to filter
            include_system: Include system namespaces (None uses config default)
            admin_only: Only return namespaces with admin access
            name_pattern: Regex pattern for namespace names
            
        Returns:
            List[NamespaceInfo]: Filtered list of namespaces
        """
        # Method stub - will be implemented in subsequent tasks
        self.logger.debug("filter_namespaces called with admin_only=%s, pattern=%s", 
                         admin_only, name_pattern)
        return namespaces
    
    def get_namespace_info(self, namespace_name: str) -> Optional[NamespaceInfo]:
        """
        Get detailed information for a specific namespace.
        
        Args:
            namespace_name: Name of the namespace to get info for
            
        Returns:
            Optional[NamespaceInfo]: Namespace information or None if not found
        """
        # Method stub - will be implemented in subsequent tasks
        self.logger.debug("get_namespace_info called for namespace=%s", namespace_name)
        return None
    
    def refresh_namespace_permissions(self, namespace: str) -> bool:
        """
        Force refresh of RBAC permissions for a specific namespace.
        
        Args:
            namespace: Namespace to refresh permissions for
            
        Returns:
            bool: True if refresh successful, False otherwise
        """
        # Method stub - will be implemented in subsequent tasks
        self.logger.debug("refresh_namespace_permissions called for namespace=%s", namespace)
        return False
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring and debugging.
        
        Returns:
            Dict[str, Any]: Cache statistics including hit rates and entry counts
        """
        with self._cache_lock:
            stats = {
                'namespace_cache_entries': len(self._namespace_cache),
                'rbac_cache_entries': len(self._rbac_cache),
                'discovery_cache_valid': self._discovery_cache is not None and self._discovery_cache.is_valid(),
                'cache_enabled': self.config.enable_cache
            }
            
            # Add performance tracking if enabled
            if self.config.enable_performance_tracking:
                stats.update(self._performance_stats.copy())
                
                # Calculate hit ratios
                namespace_total = stats['namespace_cache_hits'] + stats['namespace_cache_misses']
                rbac_total = stats['rbac_cache_hits'] + stats['rbac_cache_misses']
                discovery_total = stats['discovery_cache_hits'] + stats['discovery_cache_misses']
                
                stats.update({
                    'namespace_cache_hit_ratio': stats['namespace_cache_hits'] / namespace_total if namespace_total > 0 else 0.0,
                    'rbac_cache_hit_ratio': stats['rbac_cache_hits'] / rbac_total if rbac_total > 0 else 0.0,
                    'discovery_cache_hit_ratio': stats['discovery_cache_hits'] / discovery_total if discovery_total > 0 else 0.0,
                    'overall_cache_hit_ratio': (stats['namespace_cache_hits'] + stats['rbac_cache_hits'] + stats['discovery_cache_hits']) / 
                                              (namespace_total + rbac_total + discovery_total) if (namespace_total + rbac_total + discovery_total) > 0 else 0.0
                })
            
            return stats
    
    def _track_cache_hit(self, cache_type: str) -> None:
        """Track a cache hit for performance monitoring."""
        if self.config.enable_performance_tracking and self.config.track_cache_hit_miss:
            self._performance_stats[f'{cache_type}_cache_hits'] += 1
    
    def _track_cache_miss(self, cache_type: str) -> None:
        """Track a cache miss for performance monitoring."""
        if self.config.enable_performance_tracking and self.config.track_cache_hit_miss:
            self._performance_stats[f'{cache_type}_cache_misses'] += 1
    
    def _track_operation_time(self, operation_type: str, elapsed_time: float) -> None:
        """Track operation timing for performance monitoring."""
        if self.config.enable_performance_tracking:
            total_calls_key = f'total_{operation_type}_calls'
            avg_time_key = f'average_{operation_type}_time'
            
            current_calls = self._performance_stats[total_calls_key]
            current_avg = self._performance_stats[avg_time_key]
            
            # Calculate new average using incremental formula
            new_calls = current_calls + 1
            new_avg = (current_avg * current_calls + elapsed_time) / new_calls
            
            self._performance_stats[total_calls_key] = new_calls
            self._performance_stats[avg_time_key] = new_avg
    
    def reset_performance_stats(self) -> None:
        """Reset all performance tracking statistics."""
        if self.config.enable_performance_tracking:
            with self._cache_lock:
                for key in self._performance_stats:
                    if key.endswith('_time'):
                        self._performance_stats[key] = 0.0
                    else:
                        self._performance_stats[key] = 0
                self.logger.debug("Performance statistics reset")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a formatted performance summary.
        
        Returns:
            Dict[str, Any]: Human-readable performance summary
        """
        if not self.config.enable_performance_tracking:
            return {'performance_tracking': 'disabled'}
            
        stats = self.get_cache_statistics()
        
        summary = {
            'cache_performance': {
                'overall_hit_ratio': f"{stats.get('overall_cache_hit_ratio', 0.0):.2%}",
                'namespace_hit_ratio': f"{stats.get('namespace_cache_hit_ratio', 0.0):.2%}",
                'rbac_hit_ratio': f"{stats.get('rbac_cache_hit_ratio', 0.0):.2%}",
                'discovery_hit_ratio': f"{stats.get('discovery_cache_hit_ratio', 0.0):.2%}"
            },
            'operation_performance': {
                'total_discovery_calls': stats.get('total_discovery_calls', 0),
                'average_discovery_time': f"{stats.get('average_discovery_time', 0.0):.3f}s",
                'total_rbac_calls': stats.get('total_rbac_calls', 0),
                'average_rbac_time': f"{stats.get('average_rbac_time', 0.0):.3f}s"
            },
            'cache_entries': {
                'namespace_cache': stats.get('namespace_cache_entries', 0),
                'rbac_cache': stats.get('rbac_cache_entries', 0),
                'discovery_cache_valid': stats.get('discovery_cache_valid', False)
            }
        }
        
        # Add resilience metrics if available
        if self.config.enable_performance_tracking:
            summary['resilience_metrics'] = {
                'circuit_breaker_trips': stats.get('circuit_breaker_trips', 0),
                'retry_attempts': stats.get('retry_attempts', 0),
                'graceful_degradations': stats.get('graceful_degradations', 0),
                'circuit_breaker_enabled': self.config.enable_circuit_breaker,
                'retry_logic_enabled': self.config.enable_retry_logic,
                'memory_optimization_enabled': self.config.enable_memory_optimization
            }
            
            # Add circuit breaker states if available
            if self._discovery_circuit_breaker:
                summary['resilience_metrics']['discovery_circuit_state'] = self._discovery_circuit_breaker.state.value
            if self._rbac_circuit_breaker:
                summary['resilience_metrics']['rbac_circuit_state'] = self._rbac_circuit_breaker.state.value
        
        return summary
    
    def validate_admin_permissions(self, permissions: Dict[str, bool]) -> bool:
        """
        Validate if permission set meets admin requirements.
        
        Admin access requires ALL permissions to pass (100% success rate).
        
        Args:
            permissions: Dictionary of permission results
            
        Returns:
            bool: True if ALL permissions are granted (admin access)
        """
        if not permissions:
            return False
        
        # All permissions must be granted for admin access
        return all(granted for granted in permissions.values())
    
    def _is_system_namespace(self, namespace_name: str) -> bool:
        """
        Check if namespace is a system namespace based on naming patterns.
        
        Args:
            namespace_name: Name of the namespace to check
            
        Returns:
            bool: True if namespace appears to be a system namespace
        """
        # Helper method stub - will be implemented in subsequent tasks
        return any(namespace_name.startswith(prefix) 
                  for prefix in self.config.system_namespace_prefixes)
    
    def _generate_admin_permission_checks(self, namespace: str) -> List[Tuple[str, str, str]]:
        """
        Generate list of (verb, resource, namespace) tuples for admin permission checking.
        
        Args:
            namespace: Namespace to check permissions for
            
        Returns:
            List[Tuple[str, str, str]]: List of permission checks to perform
        """
        # Helper method stub - will be implemented in subsequent tasks
        checks = []
        for verb in self.config.admin_verbs:
            for resource in self.config.admin_resources:
                checks.append((verb, resource, namespace))
        return checks
    
    def _process_namespaces_parallel(self, 
                                   raw_namespaces: List[Dict[str, Any]], 
                                   include_rbac_check: bool) -> List[NamespaceInfo]:
        """
        Process namespaces in parallel using ThreadPoolExecutor.
        
        Args:
            raw_namespaces: Raw namespace data from Kubernetes API
            include_rbac_check: Whether to include RBAC checking
            
        Returns:
            List[NamespaceInfo]: Processed namespace information
        """
        self.logger.debug("Processing %d namespaces in parallel with %d workers", 
                         len(raw_namespaces), self.config.max_concurrent_workers)
        
        namespace_infos = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_workers) as executor:
            # Submit all namespace processing tasks
            future_to_namespace = {
                executor.submit(self._process_single_namespace, ns_data, include_rbac_check): ns_data
                for ns_data in raw_namespaces
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_namespace):
                ns_data = future_to_namespace[future]
                try:
                    namespace_info = future.result(timeout=self.config.rbac_check_timeout)
                    if namespace_info:
                        namespace_infos.append(namespace_info)
                        
                except Exception as e:
                    # Log error but continue processing other namespaces
                    ns_name = ns_data.get('name', 'unknown')
                    self.logger.error("Failed to process namespace '%s': %s", ns_name, e)
                    
                    # Create a fallback NamespaceInfo for failed processing
                    fallback_info = self._create_fallback_namespace_info(ns_data)
                    if fallback_info:
                        namespace_infos.append(fallback_info)
        
        self.logger.info("Successfully processed %d namespaces", len(namespace_infos))
        return namespace_infos
    
    def _process_single_namespace(self, 
                                ns_data: Dict[str, Any], 
                                include_rbac_check: bool) -> Optional[NamespaceInfo]:
        """
        Process a single namespace, converting raw data to NamespaceInfo with optional RBAC.
        
        Args:
            ns_data: Raw namespace data from Kubernetes API
            include_rbac_check: Whether to include RBAC checking
            
        Returns:
            Optional[NamespaceInfo]: Processed namespace information or None if failed
        """
        try:
            namespace_name = ns_data.get('name', '')
            if not namespace_name:
                self.logger.warning("Skipping namespace with missing name: %s", ns_data)
                return None
            
            # Create basic NamespaceInfo from raw data
            namespace_info = NamespaceInfo(
                name=namespace_name,
                status=ns_data.get('status', 'Unknown'),
                created=ns_data.get('created'),
                labels=ns_data.get('labels'),
                annotations=ns_data.get('annotations'),
                resource_version=ns_data.get('resource_version'),
                uid=ns_data.get('uid'),
                last_checked=time.time()
            )
            
            # Perform RBAC checking if requested
            if include_rbac_check:
                has_admin_access, admin_permissions = self.check_rbac_access(
                    namespace_name, use_cache=True
                )
                namespace_info.has_admin_access = has_admin_access
                namespace_info.admin_permissions = admin_permissions
            else:
                namespace_info.has_admin_access = False
                namespace_info.admin_permissions = {}
            
            return namespace_info
            
        except Exception as e:
            ns_name = ns_data.get('name', 'unknown')
            self.logger.error("Error processing namespace '%s': %s", ns_name, e)
            return None
    
    
    def _create_fallback_namespace_info(self, ns_data: Dict[str, Any]) -> Optional[NamespaceInfo]:
        """
        Create a fallback NamespaceInfo when processing fails.
        
        Args:
            ns_data: Raw namespace data
            
        Returns:
            Optional[NamespaceInfo]: Fallback namespace info or None if name missing
        """
        namespace_name = ns_data.get('name')
        if not namespace_name:
            return None
        
        return NamespaceInfo(
            name=namespace_name,
            status='Unknown',
            has_admin_access=False,
            admin_permissions={},
            last_checked=time.time()
        )
    
    def _cache_discovery_results(self, namespace_infos: List[NamespaceInfo]) -> None:
        """
        Cache the namespace discovery results.
        
        Args:
            namespace_infos: List of namespace information to cache
        """
        if not self.config.enable_cache:
            return
        
        try:
            with self._cache_lock:
                self._discovery_cache = CacheEntry(
                    data=namespace_infos.copy(),
                    timestamp=time.time(),
                    ttl=self.config.cache_ttl
                )
                
                # Also cache individual namespaces for future single lookups
                for ns_info in namespace_infos:
                    cache_key = f"namespace:{ns_info.name}"
                    self._namespace_cache[cache_key] = CacheEntry(
                        data=ns_info,
                        timestamp=time.time(),
                        ttl=self.config.cache_ttl
                    )
                
                self.logger.debug("Cached discovery results: %d namespaces", len(namespace_infos))
                
        except Exception as e:
            self.logger.error("Failed to cache discovery results: %s", e)
    
    def _get_cached_rbac_result(self, namespace: str) -> Optional[Tuple[bool, Dict[str, bool]]]:
        """
        Get cached RBAC result for a namespace if valid.
        
        Args:
            namespace: Namespace to get cached RBAC result for
            
        Returns:
            Optional[Tuple[bool, Dict[str, bool]]]: (has_admin_access, detailed_permissions) or None if not cached/invalid
        """
        if not self.config.enable_cache:
            return None
            
        try:
            with self._cache_lock:
                cache_key = f"rbac:{namespace}"
                
                if cache_key not in self._rbac_cache:
                    self._track_cache_miss('rbac')
                    return None
                
                cache_entry = self._rbac_cache[cache_key]
                
                if not cache_entry.is_valid():
                    # Remove expired entry
                    del self._rbac_cache[cache_key]
                    self._track_cache_miss('rbac')
                    self.logger.debug("Removed expired RBAC cache entry for namespace '%s'", namespace)
                    return None
                
                # Return cached result
                self._track_cache_hit('rbac')
                has_admin_access, detailed_permissions = cache_entry.data
                self.logger.debug("RBAC cache hit for namespace '%s': admin=%s", namespace, has_admin_access)
                return has_admin_access, detailed_permissions
                
        except Exception as e:
            self.logger.error("Error accessing RBAC cache for namespace '%s': %s", namespace, e)
            return None
    
    def _cache_rbac_result(self, namespace: str, has_admin_access: bool, detailed_permissions: Dict[str, bool]) -> None:
        """
        Cache RBAC result for a namespace with configured TTL.
        
        Args:
            namespace: Namespace to cache RBAC result for
            has_admin_access: Whether user has admin access to the namespace
            detailed_permissions: Dictionary of detailed permission results
        """
        if not self.config.enable_cache:
            return
            
        try:
            with self._cache_lock:
                cache_key = f"rbac:{namespace}"
                
                self._rbac_cache[cache_key] = CacheEntry(
                    data=(has_admin_access, detailed_permissions.copy()),
                    timestamp=time.time(),
                    ttl=self.config.rbac_cache_ttl
                )
                
                self.logger.debug("Cached RBAC result for namespace '%s': admin=%s (%d permissions)", 
                                namespace, has_admin_access, len(detailed_permissions))
                
        except Exception as e:
            self.logger.error("Failed to cache RBAC result for namespace '%s': %s", namespace, e)
    
    def _execute_with_resilience(self, operation_name: str, func, circuit_breaker=None, *args, **kwargs):
        """
        Execute operation with circuit breaker and retry logic.
        
        Args:
            operation_name: Name of operation for logging/metrics
            func: Function to execute
            circuit_breaker: Circuit breaker to use (optional)
            *args, **kwargs: Arguments for the function
            
        Returns:
            Result of function execution or None on total failure
        """
        try:
            # Track performance metrics
            if self.config.enable_performance_tracking:
                self._performance_stats[f'total_{operation_name}_calls'] += 1
            
            # Choose execution strategy based on configuration
            if circuit_breaker and self.config.enable_circuit_breaker:
                if self._retry_manager and self.config.enable_retry_logic:
                    # Circuit breaker + retry
                    def resilient_func():
                        return circuit_breaker.call(func, *args, **kwargs)
                    return self._retry_manager.execute_with_retry(resilient_func)
                else:
                    # Circuit breaker only
                    return circuit_breaker.call(func, *args, **kwargs)
            elif self._retry_manager and self.config.enable_retry_logic:
                # Retry only
                return self._retry_manager.execute_with_retry(func, *args, **kwargs)
            else:
                # Direct execution
                return func(*args, **kwargs)
                
        except Exception as e:
            self.logger.error("Operation '%s' failed after all resilience attempts: %s", operation_name, e)
            
            # Track performance metrics
            if self.config.enable_performance_tracking:
                if "circuit breaker" in str(e).lower():
                    self._performance_stats['circuit_breaker_trips'] += 1
                if hasattr(self, '_retry_manager') and self._retry_manager:
                    self._performance_stats['retry_attempts'] += getattr(self._retry_manager, 'last_attempt_count', 0)
            
            raise
    
    def _process_namespaces_in_batches(self, raw_namespaces: List[Dict[str, Any]], 
                                     include_rbac_check: bool) -> List[NamespaceInfo]:
        """
        Memory-conscious processing for large clusters using batching.
        
        Args:
            raw_namespaces: Raw namespace data from Kubernetes API
            include_rbac_check: Whether to include RBAC checking
            
        Returns:
            List[NamespaceInfo]: Processed namespace information
        """
        total_namespaces = len(raw_namespaces)
        
        # Determine if we should use batching
        if (not self.config.enable_memory_optimization or 
            total_namespaces <= self.config.large_cluster_threshold):
            return self._process_namespaces_parallel(raw_namespaces, include_rbac_check)
        
        self.logger.info("Large cluster detected (%d namespaces), using batch processing with batch size %d", 
                        total_namespaces, self.config.batch_size_large_clusters)
        
        all_namespace_infos = []
        batch_size = self.config.batch_size_large_clusters
        
        for i in range(0, total_namespaces, batch_size):
            batch = raw_namespaces[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = math.ceil(total_namespaces / batch_size)
            
            self.logger.debug("Processing batch %d/%d (%d namespaces)", 
                            batch_num, total_batches, len(batch))
            
            try:
                batch_results = self._process_namespaces_parallel(batch, include_rbac_check)
                all_namespace_infos.extend(batch_results)
                
                # Optional: Small delay between batches to reduce API pressure
                if i + batch_size < total_namespaces:
                    time.sleep(0.1)
                    
            except Exception as e:
                self.logger.error("Error processing batch %d/%d: %s", batch_num, total_batches, e)
                
                # Graceful degradation: continue with other batches
                if self.config.allow_partial_rbac_failures:
                    self.logger.warning("Continuing with other batches due to graceful degradation settings")
                    if self.config.enable_performance_tracking:
                        self._performance_stats['graceful_degradations'] += 1
                    continue
                else:
                    raise
        
        self.logger.info("Batch processing completed: %d namespaces processed", len(all_namespace_infos))
        return all_namespace_infos
    
    def _get_fallback_namespace_info(self, namespace_name: str, use_cache: bool = True) -> Optional[NamespaceInfo]:
        """
        Get fallback namespace information when primary discovery fails.
        
        Args:
            namespace_name: Name of the namespace
            use_cache: Whether to check cache for fallback data
            
        Returns:
            Optional[NamespaceInfo]: Fallback namespace info or None
        """
        if not self.config.fallback_to_cached_data or not use_cache:
            return None
            
        try:
            # Try to get from individual namespace cache
            with self._cache_lock:
                cache_key = f"namespace:{namespace_name}"
                if cache_key in self._namespace_cache:
                    cache_entry = self._namespace_cache[cache_key]
                    if cache_entry.is_valid():
                        self.logger.debug("Using cached fallback data for namespace '%s'", namespace_name)
                        if self.config.enable_performance_tracking:
                            self._performance_stats['graceful_degradations'] += 1
                        return cache_entry.data
        except Exception as e:
            self.logger.debug("Failed to get fallback data for namespace '%s': %s", namespace_name, e)
            
        return None
    
    def warm_namespace_cache(self, include_rbac_check: bool = True) -> bool:
        """
        Proactively warm the namespace cache by performing discovery.
        
        Args:
            include_rbac_check: Whether to include RBAC checking in cache warming
            
        Returns:
            bool: True if cache warming was successful
        """
        try:
            self.logger.info("Starting namespace cache warming (include_rbac=%s)", include_rbac_check)
            start_time = time.time()
            
            # Perform namespace discovery which will populate caches
            namespaces = self.discover_namespaces(
                include_rbac_check=include_rbac_check, 
                force_refresh=True  # Force fresh discovery for warming
            )
            
            elapsed_time = time.time() - start_time
            self.logger.info("Namespace cache warming completed: %d namespaces in %.2fs", 
                           len(namespaces), elapsed_time)
            
            return True
            
        except Exception as e:
            self.logger.error("Error during namespace cache warming: %s", e)
            return False
    
    def warm_rbac_cache(self, namespaces: List[str] = None) -> Dict[str, bool]:
        """
        Proactively warm the RBAC cache for specified namespaces.
        
        Args:
            namespaces: List of namespace names to warm RBAC cache for, or None to discover and warm all
            
        Returns:
            Dict[str, bool]: Mapping of namespace name to cache warming success
        """
        try:
            if namespaces is None:
                # Get current namespaces from cache or discover them
                try:
                    cached_namespaces = self.get_cached_namespaces()
                    if cached_namespaces:
                        namespaces = [ns.name for ns in cached_namespaces]
                    else:
                        # Discover namespaces without RBAC to get the list
                        discovered = self.discover_namespaces(include_rbac_check=False, force_refresh=True)
                        namespaces = [ns.name for ns in discovered]
                except Exception as e:
                    self.logger.warning("Could not get namespace list for RBAC warming: %s", e)
                    return {}
            
            self.logger.info("Starting RBAC cache warming for %d namespaces", len(namespaces))
            start_time = time.time()
            results = {}
            
            # Use ThreadPoolExecutor for parallel RBAC warming
            with ThreadPoolExecutor(max_workers=self.config.max_concurrent_workers) as executor:
                
                def warm_single_rbac(namespace: str) -> Tuple[str, bool]:
                    try:
                        # Force fresh RBAC check to populate cache
                        has_admin, _ = self.check_rbac_access(namespace, use_cache=False)
                        self.logger.debug("Warmed RBAC cache for namespace '%s': admin=%s", 
                                        namespace, has_admin)
                        return namespace, True
                    except Exception as e:
                        self.logger.warning("Failed to warm RBAC cache for namespace '%s': %s", 
                                          namespace, e)
                        return namespace, False
                
                # Submit all warming tasks
                future_to_namespace = {
                    executor.submit(warm_single_rbac, ns): ns for ns in namespaces
                }
                
                # Collect results
                for future in as_completed(future_to_namespace):
                    namespace, success = future.result()
                    results[namespace] = success
            
            elapsed_time = time.time() - start_time
            successful_count = sum(1 for success in results.values() if success)
            self.logger.info("RBAC cache warming completed: %d/%d successful in %.2fs", 
                           successful_count, len(namespaces), elapsed_time)
            
            return results
            
        except Exception as e:
            self.logger.error("Error during RBAC cache warming: %s", e)
            return {}
    
    def warm_all_caches(self, force_fresh: bool = False) -> Dict[str, Any]:
        """
        Warm all caches (namespace discovery and RBAC).
        
        Args:
            force_fresh: If True, invalidate existing caches before warming
            
        Returns:
            Dict[str, Any]: Warming results with statistics
        """
        try:
            start_time = time.time()
            self.logger.info("Starting comprehensive cache warming (force_fresh=%s)", force_fresh)
            
            if force_fresh:
                self.invalidate_all_caches()
            
            # Warm namespace cache first (includes discovery)
            namespace_success = self.warm_namespace_cache(include_rbac_check=False)
            
            # Then warm RBAC cache for all discovered namespaces
            rbac_results = self.warm_rbac_cache()
            
            elapsed_time = time.time() - start_time
            
            # Compile results
            results = {
                'namespace_warming_success': namespace_success,
                'rbac_warming_results': rbac_results,
                'total_namespaces': len(rbac_results),
                'successful_rbac_warming': sum(1 for success in rbac_results.values() if success),
                'elapsed_time': elapsed_time,
                'cache_statistics': self.get_cache_statistics()
            }
            
            self.logger.info("Comprehensive cache warming completed in %.2fs: %d/%d namespaces", 
                           elapsed_time, results['successful_rbac_warming'], results['total_namespaces'])
            
            return results
            
        except Exception as e:
            self.logger.error("Error during comprehensive cache warming: %s", e)
            return {
                'namespace_warming_success': False,
                'rbac_warming_results': {},
                'error': str(e)
            }
    
    def invalidate_namespace_cache(self, namespace: str = None) -> bool:
        """
        Invalidate namespace cache entries.
        
        Args:
            namespace: Specific namespace to invalidate, or None to invalidate all namespace caches
            
        Returns:
            bool: True if any cache entries were invalidated
        """
        if not self.config.enable_cache:
            return False
            
        try:
            with self._cache_lock:
                invalidated = False
                
                if namespace is None:
                    # Clear all namespace-related caches
                    if self._discovery_cache is not None:
                        self._discovery_cache = None
                        invalidated = True
                        self.logger.debug("Invalidated discovery cache")
                    
                    if self._namespace_cache:
                        cleared_count = len(self._namespace_cache)
                        self._namespace_cache.clear()
                        invalidated = True
                        self.logger.debug("Invalidated %d namespace cache entries", cleared_count)
                        
                else:
                    # Clear specific namespace cache
                    cache_key = f"namespace:{namespace}"
                    if cache_key in self._namespace_cache:
                        del self._namespace_cache[cache_key]
                        invalidated = True
                        self.logger.debug("Invalidated cache for namespace '%s'", namespace)
                    
                    # If this was the only namespace, also clear discovery cache
                    if self._discovery_cache is not None:
                        self._discovery_cache = None
                        invalidated = True
                        self.logger.debug("Invalidated discovery cache due to namespace change")
                        
                return invalidated
                
        except Exception as e:
            self.logger.error("Error invalidating namespace cache: %s", e)
            return False
    
    def invalidate_rbac_cache(self, namespace: str = None) -> bool:
        """
        Invalidate RBAC cache entries.
        
        Args:
            namespace: Specific namespace to invalidate RBAC cache for, or None to invalidate all RBAC caches
            
        Returns:
            bool: True if any cache entries were invalidated
        """
        if not self.config.enable_cache:
            return False
            
        try:
            with self._cache_lock:
                invalidated = False
                
                if namespace is None:
                    # Clear all RBAC caches
                    if self._rbac_cache:
                        cleared_count = len(self._rbac_cache)
                        self._rbac_cache.clear()
                        invalidated = True
                        self.logger.debug("Invalidated %d RBAC cache entries", cleared_count)
                        
                else:
                    # Clear specific namespace RBAC cache
                    cache_key = f"rbac:{namespace}"
                    if cache_key in self._rbac_cache:
                        del self._rbac_cache[cache_key]
                        invalidated = True
                        self.logger.debug("Invalidated RBAC cache for namespace '%s'", namespace)
                        
                return invalidated
                
        except Exception as e:
            self.logger.error("Error invalidating RBAC cache: %s", e)
            return False
    
    def invalidate_all_caches(self) -> bool:
        """
        Invalidate all cache entries.
        
        Returns:
            bool: True if any cache entries were invalidated
        """
        try:
            namespace_invalidated = self.invalidate_namespace_cache()
            rbac_invalidated = self.invalidate_rbac_cache()
            
            if namespace_invalidated or rbac_invalidated:
                self.logger.info("Invalidated all caches")
                return True
            else:
                self.logger.debug("No cache entries to invalidate")
                return False
                
        except Exception as e:
            self.logger.error("Error invalidating all caches: %s", e)
            return False
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        try:
            self.clear_cache()
            self.logger.debug("NamespaceManager cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during NamespaceManager cleanup: {e}")