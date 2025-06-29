import streamlit as st
import sys
import os
import logging
from typing import List, Optional

# Import authentication functions
from auth import (
    check_authentication, 
    get_current_user,
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


@st.cache_resource
def get_namespace_manager(bearer_token: str) -> Optional[NamespaceManager]:
    """Initialize and cache the namespace manager with Kubernetes client."""
    try:
        # Configure Kubernetes client with bearer token from authentication
        k8s_config = K8sClientConfig(bearer_token=bearer_token)
        k8s_client = KubernetesClient(k8s_config)
        
        # Create namespace manager with optimized configuration
        ns_config = NamespaceManagerConfig(
            max_concurrent_workers=10,
            discovery_cache_ttl=300,  # 5 minutes
            rbac_cache_ttl=600,       # 10 minutes
            enable_performance_tracking=True,
            enable_circuit_breaker=True,
            enable_retry_logic=True
        )
        
        return NamespaceManager(k8s_client, ns_config)
    except Exception as e:
        logging.error(f"Failed to initialize namespace manager: {e}")
        return None


def load_namespaces(namespace_manager: NamespaceManager) -> tuple[List[NamespaceInfo], Optional[str]]:
    """Load namespaces with error handling and return (namespaces, error_message)."""
    try:
        # Discover namespaces with RBAC filtering
        namespaces = namespace_manager.discover_namespaces(include_rbac_check=True)
        
        # Sort namespaces by name (ascending, case-insensitive)
        sorted_namespaces = namespace_manager.sort_namespaces(namespaces, SortOrder.NAME_ASC)
        
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
            return
        
        # Initialize namespace manager
        namespace_manager = get_namespace_manager(user.bearer_token)
        
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
            
            # Load namespaces with caching and error handling
            if refresh_clicked or 'namespaces_data' not in st.session_state:
                with st.spinner("Loading namespaces..."):
                    namespaces, error_msg = load_namespaces(namespace_manager)
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