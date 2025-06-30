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
        st.subheader("🔄 Restore Section")
        
        # Check if user can perform restore operations
        if require_groups(["admin", "backup-admin", "restore-admin"], user):
            st.info("You have permission to perform restore operations.")
            if st.button("View Restore History", disabled=True):
                st.info("Restore functionality will be available once Velero integration is complete")
        else:
            st.warning("You don't have permission to perform restore operations.")
            st.caption("Required groups: admin, backup-admin, or restore-admin")
    
    with col2:
        st.subheader("💾 Backup Section")
        
        # Check if user can perform backup operations  
        if require_groups(["admin", "backup-admin"], user):
            st.info("You have permission to perform backup operations.")
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