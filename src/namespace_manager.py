"""
Namespace Manager module for Velero Manager application.

This module provides namespace discovery with RBAC filtering capabilities,
leveraging multithreaded operations for efficient namespace management.
It includes admin permission checking, caching, and sorting functionality.
"""

import logging
import time
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
    
    # Minimum admin permissions required (percentage)
    admin_threshold: float = 0.6  # 60% of admin permissions required
    
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
            # Step 2: Fetch raw namespace data from Kubernetes API
            self.logger.debug("Fetching namespaces from Kubernetes API")
            raw_namespaces = self.k8s_client.list_namespaces(use_cache=not force_refresh)
            
            if not raw_namespaces:
                self.logger.warning("No namespaces found from Kubernetes API")
                return []
            
            self.logger.info("Fetched %d raw namespaces from API", len(raw_namespaces))
            
            # Step 3: Process namespaces in parallel using ThreadPoolExecutor
            namespace_infos = self._process_namespaces_parallel(
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
        
        Args:
            namespace: Namespace to check permissions for
            use_cache: Whether to use cached results
            
        Returns:
            Tuple[bool, Dict[str, bool]]: (has_admin_access, detailed_permissions)
        """
        # Method stub - will be implemented in subsequent tasks
        self.logger.info("check_rbac_access called for namespace=%s, use_cache=%s", 
                        namespace, use_cache)
        return False, {}
    
    def get_cached_namespaces(self) -> Optional[List[NamespaceInfo]]:
        """
        Get cached namespace discovery results if valid.
        
        Returns:
            Optional[List[NamespaceInfo]]: Cached results or None if invalid/missing
        """
        # Method stub - will be implemented in subsequent tasks
        with self._cache_lock:
            if not self.config.enable_cache or not self._discovery_cache:
                return None
            
            if self._discovery_cache.is_valid():
                self.logger.debug("Returning cached namespace discovery results")
                return self._discovery_cache.data
            
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
        # Method stub - will be implemented in subsequent tasks
        order = sort_order or self.config.default_sort_order
        self.logger.debug("sort_namespaces called with order=%s for %d namespaces", 
                         order, len(namespaces))
        return namespaces
    
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
        # Method stub - will be implemented in subsequent tasks
        with self._cache_lock:
            return {
                'namespace_cache_entries': len(self._namespace_cache),
                'rbac_cache_entries': len(self._rbac_cache),
                'discovery_cache_valid': self._discovery_cache is not None and self._discovery_cache.is_valid(),
                'cache_enabled': self.config.enable_cache
            }
    
    def validate_admin_permissions(self, permissions: Dict[str, bool]) -> bool:
        """
        Validate if permission set meets admin threshold.
        
        Args:
            permissions: Dictionary of permission results
            
        Returns:
            bool: True if permissions meet admin threshold
        """
        # Method stub - will be implemented in subsequent tasks
        if not permissions:
            return False
        
        granted_count = sum(1 for granted in permissions.values() if granted)
        total_count = len(permissions)
        
        if total_count == 0:
            return False
        
        percentage = granted_count / total_count
        return percentage >= self.config.admin_threshold
    
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
                has_admin_access, admin_permissions = self._check_namespace_rbac_placeholder(
                    namespace_name
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
    
    def _check_namespace_rbac_placeholder(self, namespace: str) -> Tuple[bool, Dict[str, bool]]:
        """
        Placeholder RBAC checking method for Task 4.2.
        
        This will be replaced with real SelfSubjectAccessReview logic in Task 4.3.
        For now, returns admin access for namespaces starting with "openshift-".
        
        Args:
            namespace: Namespace to check permissions for
            
        Returns:
            Tuple[bool, Dict[str, bool]]: (has_admin_access, detailed_permissions)
        """
        # Placeholder logic: openshift- namespaces have admin access
        has_admin = namespace.startswith('openshift-')
        
        # Generate placeholder permission details
        permissions = {}
        for verb in self.config.admin_verbs:
            for resource in self.config.admin_resources:
                # OpenShift namespaces get admin permissions, others don't
                permissions[f"{verb}:{resource}"] = has_admin
        
        self.logger.debug("Placeholder RBAC check for '%s': admin=%s", namespace, has_admin)
        return has_admin, permissions
    
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