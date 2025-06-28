import streamlit as st
import sys
import os

# Import authentication functions
from auth import (
    check_authentication, 
    get_current_user,
    require_groups,
    is_user_admin,
    set_dev_headers,
    clear_dev_headers
)

st.set_page_config(
    page_title="Velero Manager",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
        # For now, show placeholder - will be replaced with actual namespace discovery
        st.selectbox("Select Namespace", ["Loading..."], disabled=True, 
                    help="Namespace selection will be available once Kubernetes API integration is complete")
    
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