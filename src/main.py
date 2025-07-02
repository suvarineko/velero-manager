import streamlit as st
import os
import logging
from typing import List, Optional

# Import authentication functions
from auth import (
    check_authentication, 
    require_groups,
    is_user_admin,
    set_dev_headers,
    clear_dev_headers
)

# Import Kubernetes and namespace management
from k8s_client import KubernetesClient, K8sClientConfig
from namespace_manager import NamespaceManager, NamespaceManagerConfig, NamespaceInfo, SortOrder

# Import UI components
from restore_ui import display_restore_ui

st.set_page_config(
    page_title="Velero Manager",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@st.cache_resource
def get_namespace_manager(user_info) -> Optional[NamespaceManager]:
    """Initialize and cache the namespace manager with Kubernetes client."""
    try:
        # Configure Kubernetes client
        k8s_config = K8sClientConfig(
            connection_timeout=30,
            read_timeout=60,
            max_retries=3,
            enable_circuit_breaker=True,
            resource_cache_ttl=300,  # 5 minutes
            namespace_cache_ttl=600  # 10 minutes
        )
        k8s_client = KubernetesClient(k8s_config)
        
        # Authenticate with user info (proper method that sets up session)
        logging.info(f"Attempting authentication with token length: {len(user_info.bearer_token) if user_info.bearer_token else 0}")
        
        auth_result = k8s_client.authenticate_with_user_info(user_info)
        logging.info(f"Authentication result: {auth_result}")
        
        if not auth_result:
            logging.error("Failed to authenticate with Kubernetes API")
            return None
        
        # Verify authentication worked
        if not k8s_client.is_authenticated():
            logging.error("Authentication completed but client is not authenticated")
            logging.error(f"Current user set: {k8s_client._current_user is not None}")
            logging.error(f"API client set: {k8s_client.api_client is not None}")
            logging.error(f"Session valid: {k8s_client._is_session_valid()}")
            return None
        
        logging.info("Kubernetes client successfully authenticated and session established")
        
        # Create namespace manager with optimized configuration
        ns_config = NamespaceManagerConfig(
            max_concurrent_workers=10,
            cache_ttl=300.0,          # 5 minutes for namespace discovery
            rbac_cache_ttl=600.0,     # 10 minutes for RBAC results
            enable_performance_tracking=True,
            enable_circuit_breaker=True,
            enable_retry_logic=True,
            enable_memory_optimization=True,
            large_cluster_threshold=500,
            batch_size_large_clusters=50
        )
        
        return NamespaceManager(k8s_client, ns_config)
    except Exception as e:
        logging.error(f"Failed to initialize namespace manager: {e}")
        return None


def load_namespaces(namespace_manager: NamespaceManager, force_refresh: bool = False) -> tuple[List[NamespaceInfo], Optional[str]]:
    """
    Load namespaces with session-level caching and error handling.
    
    Args:
        namespace_manager: The NamespaceManager instance
        force_refresh: If True, bypass all caches and fetch fresh data
                      If False, use session-level cached data if available (600s TTL)
    """
    import time
    
    # Session cache configuration
    SESSION_CACHE_TTL = 600  # 600 seconds as requested
    
    try:
        # Check session-level cache first (unless force refresh)
        if not force_refresh:
            # Check if we have valid cached data in session state
            if ('namespace_cache_data' in st.session_state and 
                'namespace_cache_timestamp' in st.session_state):
                
                cache_age = time.time() - st.session_state.namespace_cache_timestamp
                if cache_age < SESSION_CACHE_TTL:
                    logging.info(f"Using session-cached namespaces (age: {cache_age:.1f}s, TTL: {SESSION_CACHE_TTL}s)")
                    return st.session_state.namespace_cache_data, None
                else:
                    logging.info(f"Session cache expired (age: {cache_age:.1f}s, TTL: {SESSION_CACHE_TTL}s)")
            else:
                logging.info("No session cache found, fetching fresh data")
        else:
            logging.info("Force refresh requested, bypassing all caches")
        
        # Fetch from NamespaceManager (which may use its own internal cache)
        namespaces = namespace_manager.discover_namespaces(
            include_rbac_check=True,
            force_refresh=force_refresh
        )
        
        # Sort namespaces by name (ascending, case-insensitive)
        sorted_namespaces = namespace_manager.sort_namespaces(namespaces, SortOrder.NAME_ASC)
        
        # Store in session cache for future page reloads
        st.session_state.namespace_cache_data = sorted_namespaces
        st.session_state.namespace_cache_timestamp = time.time()
        logging.info(f"Stored {len(sorted_namespaces)} namespaces in session cache")
        
        return sorted_namespaces, None
        
    except Exception as e:
        error_msg = f"Failed to load namespaces: {str(e)}"
        logging.error(error_msg)
        return [], error_msg


def display_namespace_resources(namespace_manager: NamespaceManager, namespace: str):
    """
    Display a tree view of all resources in the selected namespace organized by Kind.
    
    Args:
        namespace_manager: The NamespaceManager instance
        namespace: The namespace to discover resources in
    """
    try:
        # Get the k8s_client from namespace_manager
        k8s_client = namespace_manager.k8s_client
        
        # Show loading state while discovering resources
        with st.spinner(f"Discovering resources in namespace '{namespace}'..."):
            # Discover all resources in the namespace (use cache=False for completeness)
            resources = k8s_client.discover_namespace_resources(
                namespace=namespace,
                use_cache=True,  # Use cache for better performance
                include_crds=True  # Include custom resources
            )
        
        # Check if namespace is empty
        if not resources:
            st.info(f"📁 Namespace '{namespace}' doesn't contain any resources")
            st.caption("This namespace appears to be empty or you may not have permissions to view resources.")
            return
        
        # Display resources organized by Kind
        st.markdown("#### 📦 Available Resources")
        st.caption(f"Resources found in namespace '{namespace}' organized by type")
        
        # Group resources by Kind and count them
        total_resources = 0
        kind_counts = {}
        
        for resource_type, resource_list in resources.items():
            # Extract Kind from resource_type (e.g., "v1/pods" -> "Pod")
            kind = _extract_kind_from_resource_type(resource_type)
            
            if kind not in kind_counts:
                kind_counts[kind] = []
            
            if resource_list:  # Non-empty resource types
                # Add all resources of this type to the kind
                kind_counts[kind].extend(resource_list)
                total_resources += len(resource_list)
        
        # Display total count (only count kinds with resources)
        non_empty_kinds = len([k for k, v in kind_counts.items() if v])
        st.caption(f"Total: {total_resources} resources across {non_empty_kinds} resource types")
        
        # Display custom HTML/CSS tree component
        _display_resource_tree_html(kind_counts)
    
    except Exception as e:
        # Handle permission errors gracefully
        error_str = str(e).lower()
        if "403" in error_str or "forbidden" in error_str or "permission" in error_str:
            st.warning(f"⚠️ Permission Error")
            st.caption(f"You don't have sufficient permissions to view resources in namespace '{namespace}'")
        else:
            st.error(f"❌ Error discovering resources: {str(e)}")
            st.caption("Please check your connection and permissions")


def _extract_kind_from_resource_type(resource_type: str) -> str:
    """
    Extract Kubernetes Kind from resource type string.
    
    Args:
        resource_type: Resource type like "v1/pods", "apps/v1/deployments"
        
    Returns:
        str: The Kubernetes Kind like "Pod", "Deployment"
    """
    try:
        # Handle different resource type formats
        if '/' in resource_type:
            # Extract the last part (resource name) and convert to Kind
            resource_name = resource_type.split('/')[-1]
        else:
            resource_name = resource_type
        
        # Convert resource name to Kind (singular, capitalized)
        kind_mapping = {
            'pods': 'Pod',
            'services': 'Service', 
            'deployments': 'Deployment',
            'replicasets': 'ReplicaSet',
            'statefulsets': 'StatefulSet',
            'daemonsets': 'DaemonSet',
            'jobs': 'Job',
            'cronjobs': 'CronJob',
            'configmaps': 'ConfigMap',
            'secrets': 'Secret',
            'persistentvolumeclaims': 'PersistentVolumeClaim',
            'serviceaccounts': 'ServiceAccount',
            'endpoints': 'Endpoints',
            'events': 'Event',
            'ingresses': 'Ingress',
            'networkpolicies': 'NetworkPolicy',
            'roles': 'Role',
            'rolebindings': 'RoleBinding',
            'storageclasses': 'StorageClass'
        }
        
        # Return mapped kind or capitalize the resource name
        return kind_mapping.get(resource_name.lower(), resource_name.capitalize())
        
    except Exception:
        return resource_type  # Fallback to original if parsing fails


def _display_resource_tree_html(kind_counts: dict):
    """
    Display resources using a custom HTML/CSS tree component.
    
    Args:
        kind_counts: Dictionary of Kind -> list of resources
    """
    import streamlit.components.v1 as components
    
    # Build HTML structure with embedded CSS and JavaScript
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
        .resource-tree {{
            font-family: "Source Sans Pro", sans-serif;
            margin: 10px 0;
            border: 1px solid #e6e6e6;
            border-radius: 6px;
            background: #fafafa;
            padding: 5px;
        }}
        
        .tree-node {{
            margin: 2px 0;
            border-radius: 4px;
            overflow: hidden;
            border: 1px solid #e0e0e0;
            background: white;
        }}
        
        .tree-header {{
            display: flex;
            align-items: center;
            padding: 8px 12px;
            cursor: pointer;
            background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
            border-bottom: 1px solid #e0e0e0;
            user-select: none;
            transition: background-color 0.2s ease;
        }}
        
        .tree-header:hover {{
            background: linear-gradient(90deg, #e9ecef 0%, #dee2e6 100%);
        }}
        
        .tree-icon {{
            margin-right: 8px;
            font-size: 12px;
            transition: transform 0.2s ease;
            color: #666;
        }}
        
        .tree-icon.expanded {{
            transform: rotate(90deg);
        }}
        
        .tree-title {{
            font-weight: 600;
            color: #262730;
            font-size: 14px;
        }}
        
        .tree-count {{
            margin-left: auto;
            background: #007acc;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }}
        
        .tree-content {{
            display: none;
            padding: 8px 16px 12px 16px;
            background: #fdfdfd;
            border-top: 1px solid #f0f0f0;
        }}
        
        .tree-content.expanded {{
            display: block;
        }}
        
        .resource-item {{
            padding: 3px 0;
            color: #555;
            font-size: 13px;
            display: flex;
            align-items: center;
        }}
        
        .resource-item:before {{
            content: "•";
            color: #007acc;
            margin-right: 8px;
            font-weight: bold;
        }}
        
        .empty-message {{
            color: #888;
            font-style: italic;
            font-size: 13px;
            padding: 20px;
            text-align: center;
        }}
        </style>
    </head>
    <body>
        <div class="resource-tree">
    """
    
    # Add each Kind as a tree node
    if not any(kind_counts.values()):
        html_content += '<div class="empty-message">No resources found in this namespace</div>'
    else:
        for kind in sorted(kind_counts.keys()):
            resource_list = kind_counts[kind]
            
            # Only show Kinds that have resources
            if resource_list:
                # Sort resources by name
                sorted_resources = sorted(resource_list, key=lambda x: x.get('name', ''))
                # Sanitize kind name for use as ID (remove special characters)
                kind_id = ''.join(c for c in kind if c.isalnum())
                
                html_content += f"""
                <div class="tree-node">
                    <div class="tree-header" onclick="toggleNode('{kind_id}')">
                        <span class="tree-icon" id="icon-{kind_id}">▶</span>
                        <span class="tree-title">📋 {kind}</span>
                        <span class="tree-count">{len(resource_list)}</span>
                    </div>
                    <div class="tree-content" id="content-{kind_id}">
                """
                
                # Add resource items
                for resource in sorted_resources:
                    resource_name = resource.get('name', 'Unknown')
                    # Escape HTML special characters
                    resource_name = resource_name.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    html_content += f'<div class="resource-item">{resource_name}</div>'
                
                html_content += """
                    </div>
                </div>
                """
    
    html_content += """
        </div>
        
        <script>
        function toggleNode(kindId) {
            const icon = document.getElementById('icon-' + kindId);
            const content = document.getElementById('content-' + kindId);
            
            if (content.classList.contains('expanded')) {
                content.classList.remove('expanded');
                icon.classList.remove('expanded');
                icon.innerHTML = '▶';
            } else {
                content.classList.add('expanded');
                icon.classList.add('expanded');
                icon.innerHTML = '▼';
            }
        }
        </script>
    </body>
    </html>
    """
    
    # Display using Streamlit components
    components.html(html_content, height=400, scrolling=True)


def main():
    try:
        st.title("🔄 Velero Manager")
        st.markdown("### Kubernetes Backup and Restore Management")
        
        # Check authentication - required for the entire application
        user = check_authentication(show_error=False)
        
        if not user:
            # Show authentication required page
            show_authentication_page()
            return
        
        # Show authenticated interface
        show_main_interface(user)
            
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please check the logs for more details.")


def show_authentication_page():
    """Display authentication required page with development options."""
    st.warning("🔒 Authentication Required")
    st.info("Please authenticate through the OAuth proxy to access Velero Manager.")
    
    # Show development mode options
    if os.getenv('DEV_MODE', '').lower() == 'true':
        st.markdown("---")
        st.subheader("🧪 Development Mode")
        st.info("Development mode is enabled. You can set test authentication headers below:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Quick Setup:**")
            if st.button("Set Admin User", key="admin_auth"):
                set_dev_headers("admin", "Administrator", ["admin", "cluster-admin"], "admin-token")
                st.success("Admin authentication set! Refreshing...")
                st.rerun()
            
            if st.button("Set Backup Admin", key="backup_auth"):
                set_dev_headers("backup-admin", "Backup Admin", ["backup-admin", "user"], "backup-token")
                st.success("Backup admin authentication set! Refreshing...")
                st.rerun()
            
            if st.button("Set Regular User", key="user_auth"):
                set_dev_headers("user", "Regular User", ["user", "readonly"], "user-token")
                st.success("User authentication set! Refreshing...")
                st.rerun()
        
        with col2:
            st.markdown("**Custom Setup:**")
            username = st.text_input("Username", value="testuser")
            display_name = st.text_input("Display Name", value="Test User")
            groups = st.text_input("Groups (comma-separated)", value="user,backup-admin")
            token = st.text_input("Bearer Token", value="test-token-123")
            
            if st.button("Set Custom Auth", key="custom_auth"):
                group_list = [g.strip() for g in groups.split(',') if g.strip()]
                set_dev_headers(username, display_name, group_list, token)
                st.success("Custom authentication set! Refreshing...")
                st.rerun()


def show_main_interface(user):
    """Display the main application interface for authenticated users."""
    # Header with user info
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("Welcome to Velero Manager - your centralized interface for managing Velero backup and restore operations.")
    
    with col2:
        st.markdown(f"**👤 {user.preferred_username}**")
        st.caption(f"Groups: {', '.join(user.groups) if user.groups else 'None'}")
        if st.button("Logout", key="logout"):
            from auth import clear_session
            clear_session()
            st.success("Logged out successfully!")
            st.rerun()
    
    st.markdown("---")
    
    # Namespace selection and user info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Namespace Selection")
        
        # Check for bearer token
        if not hasattr(user, 'bearer_token') or not user.bearer_token:
            st.error("❌ Missing authentication token")
            st.caption("Bearer token is required for Kubernetes API access. Please re-authenticate.")
            # Debug info in development mode
            if os.getenv('DEV_MODE', '').lower() == 'true':
                st.code(f"User object: {type(user)}")
                st.code(f"User attributes: {dir(user)}")
                if hasattr(user, 'raw_headers'):
                    st.code(f"Raw headers keys: {list(user.raw_headers.keys()) if user.raw_headers else 'None'}")
            return
        
        # Debug token info in development mode
        if os.getenv('DEV_MODE', '').lower() == 'true':
            logging.info(f"User bearer token available: {bool(user.bearer_token)}")
            logging.info(f"Token starts with: {user.bearer_token[:20] if user.bearer_token else 'None'}...")
        
        # Initialize namespace manager
        namespace_manager = get_namespace_manager(user)
        
        if not namespace_manager:
            st.error("❌ Failed to initialize Kubernetes connection")
            st.caption("Please check your authentication and cluster connectivity")
        else:
            # Initialize session state for selected namespace
            if 'selected_namespace' not in st.session_state:
                st.session_state.selected_namespace = None
            
            # Create refresh button
            col_ns1, col_ns2 = st.columns([3, 1])
            
            with col_ns1:
                st.write("")  # Empty space for alignment
            
            with col_ns2:
                refresh_clicked = st.button("🔄", help="Refresh namespace list", key="refresh_ns")
            
            # Load namespaces with session-level caching
            # Always load namespaces, let the load_namespaces function handle caching logic
            if refresh_clicked:
                # Manual refresh - bypass all caches to get fresh data
                with st.spinner("Refreshing namespaces..."):
                    namespaces, error_msg = load_namespaces(namespace_manager, force_refresh=True)
            else:
                # Page reload/automatic load - use session cache if available (600s TTL)
                with st.spinner("Loading namespaces..."):
                    namespaces, error_msg = load_namespaces(namespace_manager, force_refresh=False)
            
            # Store results for UI components (maintains backward compatibility)
            st.session_state.namespaces_data = namespaces
            st.session_state.namespaces_error = error_msg
            
            # Display namespace selection
            if st.session_state.get('namespaces_error'):
                st.error(f"❌ {st.session_state.namespaces_error}")
                st.caption("Unable to load namespaces. Please check your permissions and cluster connectivity.")
            elif not st.session_state.get('namespaces_data'):
                st.info("🔍 No namespaces available or none found with current permissions")
            else:
                namespaces = st.session_state.namespaces_data
                
                # Create namespace options with admin access indicators
                namespace_options = []
                namespace_labels = []
                
                for ns in namespaces:
                    namespace_options.append(ns.name)
                    admin_indicator = " 🔑" if ns.has_admin_access else ""
                    namespace_labels.append(f"{ns.name}{admin_indicator}")
                
                # Find current selection index
                current_index = 0
                if st.session_state.selected_namespace and st.session_state.selected_namespace in namespace_options:
                    current_index = namespace_options.index(st.session_state.selected_namespace)
                
                # Namespace selection dropdown
                selected_ns = st.selectbox(
                    "Select Namespace",
                    options=namespace_options,
                    format_func=lambda x: next((label for ns, label in zip(namespace_options, namespace_labels) if ns == x), x),
                    index=current_index,
                    help="🔑 indicates admin access to the namespace",
                    key="namespace_selector"
                )
                
                # Update session state
                if selected_ns != st.session_state.selected_namespace:
                    st.session_state.selected_namespace = selected_ns
                
                # Show namespace info
                if selected_ns:
                    selected_ns_info = next((ns for ns in namespaces if ns.name == selected_ns), None)
                    if selected_ns_info:
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.caption(f"Status: {selected_ns_info.status}")
                        with col_info2:
                            access_level = "Admin" if selected_ns_info.has_admin_access else "Read-only"
                            st.caption(f"Access: {access_level}")
                
                # Show statistics
                total_ns = len(namespaces)
                admin_ns = sum(1 for ns in namespaces if ns.has_admin_access)
                st.caption(f"Total: {total_ns} namespaces ({admin_ns} with admin access)")
    
    with col2:
        st.subheader("👤 User Details")
        st.text(f"Username: {user.username}")
        st.text(f"Display Name: {user.preferred_username}")
        if user.groups:
            st.text(f"Groups: {', '.join(user.groups)}")
        st.text(f"Admin: {'Yes' if is_user_admin(user) else 'No'}")
    
    st.markdown("---")
    
    # Main content sections
    col1, col2 = st.columns(2)
    
    with col1:
        # Display restore UI component with BackupManager integration
        if namespace_manager and st.session_state.get('selected_namespace'):
            # Get the k8s_client from namespace_manager for restore_ui
            k8s_client = namespace_manager.k8s_client
            
            # Display the restore UI component
            display_restore_ui(
                selected_namespace=st.session_state.selected_namespace,
                k8s_client=k8s_client,
                user_permissions=user.groups if user.groups else []
            )
        else:
            # Fallback when namespace or client not available
            st.subheader("🔄 Restore Section")
            if not namespace_manager:
                st.error("❌ Kubernetes client not available")
                st.caption("Unable to connect to Kubernetes API. Please check authentication.")
            elif not st.session_state.get('selected_namespace'):
                st.info("📁 Please select a namespace to view available backups")
                st.caption("Use the namespace dropdown above to select a namespace.")
    
    with col2:
        st.subheader("💾 Backup Section")
        
        # Check if user can perform backup operations  
        if require_groups(["system:authenticated"], user):
            st.info("You have permission to perform backup operations.")
            
            # Show resource tree for selected namespace
            if namespace_manager and st.session_state.get('selected_namespace'):
                display_namespace_resources(namespace_manager, st.session_state.selected_namespace)
            elif not st.session_state.get('selected_namespace'):
                st.info("📁 Please select a namespace to view resources for backup")
                st.caption("Use the namespace dropdown above to select a namespace.")
            
            if st.button("Create Backup", disabled=True):
                st.info("Backup functionality will be available once Velero integration is complete")
        else:
            st.warning("You don't have permission to perform backup operations.")
            st.caption("Required groups: admin or backup-admin")
    
    # Show development options if in dev mode
    if os.getenv('DEV_MODE', '').lower() == 'true':
        with st.expander("🧪 Development Options"):
            st.info("Development mode is active")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Switch to Admin"):
                    set_dev_headers("admin", "Administrator", ["admin", "cluster-admin"], "admin-token")
                    st.rerun()
                
                if st.button("Switch to Backup Admin"):
                    set_dev_headers("backup-admin", "Backup Admin", ["backup-admin", "user"], "backup-token")
                    st.rerun()
            
            with col2:
                if st.button("Switch to Regular User"):
                    set_dev_headers("user", "Regular User", ["user", "readonly"], "user-token")
                    st.rerun()
                
                if st.button("Clear Auth Headers"):
                    clear_dev_headers()
                    st.rerun()


if __name__ == "__main__":
    main()