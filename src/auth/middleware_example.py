"""
Example usage of authentication decorators and middleware.

This file demonstrates how to use the authentication decorators and middleware
in a Streamlit application with various access control scenarios.
"""

import streamlit as st
from typing import List, Dict, Any

# Import authentication functions
from . import (
    require_auth, 
    check_authentication, 
    require_groups,
    get_user_groups,
    is_user_admin,
    set_dev_headers,
    clear_dev_headers
)


# Example 1: Function-level protection with decorator
@require_auth
def get_user_info() -> Dict[str, Any]:
    """Function that requires authentication but no specific groups."""
    from . import get_current_user
    user = get_current_user()
    return {
        "username": user.username,
        "preferred_username": user.preferred_username,
        "groups": user.groups
    }


@require_auth(groups=["admin"])
def get_admin_data() -> str:
    """Function that requires admin group membership."""
    return "This is sensitive admin data that only admins can see"


@require_auth(groups=["backup-admin", "cluster-admin"])
def manage_backups() -> str:
    """Function that requires backup or cluster admin privileges."""
    return "Backup management functionality available"


@require_auth(groups=["readonly", "user", "admin"])
def view_backup_status() -> str:
    """Function that allows multiple group types."""
    return "Backup status information (read-only access)"


# Example 2: Streamlit page with middleware protection
def protected_page_example():
    """Example of a Streamlit page using check_authentication middleware."""
    st.title("🔒 Protected Page Example")
    
    # Check authentication at page level
    user = check_authentication()
    if not user:
        return  # Authentication failed, error already displayed
    
    st.success(f"Welcome, {user.preferred_username}!")
    st.info(f"Username: {user.username}")
    st.info(f"Groups: {', '.join(user.groups) if user.groups else 'None'}")
    
    # Show different content based on user groups
    if is_user_admin(user):
        st.subheader("🛠️ Admin Functions")
        st.write("You have admin privileges!")
        
        if st.button("Get Admin Data"):
            try:
                data = get_admin_data()
                st.success(data)
            except Exception as e:
                st.error(f"Error: {e}")
    
    if require_groups(["backup-admin", "cluster-admin"], user):
        st.subheader("💾 Backup Management")
        st.write("You can manage backups!")
        
        if st.button("Access Backup Functions"):
            try:
                result = manage_backups()
                st.success(result)
            except Exception as e:
                st.error(f"Error: {e}")
    
    # This should work for most users
    st.subheader("📊 Backup Status")
    if st.button("View Backup Status"):
        try:
            status = view_backup_status()
            st.info(status)
        except Exception as e:
            st.error(f"Error: {e}")


def admin_only_page():
    """Example of a page that requires admin access."""
    st.title("👑 Admin Only Page")
    
    # Check authentication with admin requirement
    user = check_authentication(required_groups=["admin", "cluster-admin"])
    if not user:
        return  # Access denied, error already displayed
    
    st.success("Welcome to the admin area!")
    st.write("Only administrators can see this content.")
    
    # Admin-specific functionality
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("User Management")
        st.button("Manage Users")
        st.button("View Audit Logs")
    
    with col2:
        st.subheader("System Configuration")
        st.button("Configure Settings")
        st.button("Manage Permissions")


def development_testing_page():
    """Page for testing authentication in development mode."""
    st.title("🧪 Development Testing")
    st.write("This page helps test authentication functionality in development mode.")
    
    # Show current authentication status
    user = check_authentication(show_error=False)
    
    if user:
        st.success(f"✅ Authenticated as: {user.preferred_username}")
        st.info(f"Username: {user.username}")
        st.info(f"Groups: {', '.join(user.groups) if user.groups else 'None'}")
        st.info(f"Has Bearer Token: {'Yes' if user.bearer_token else 'No'}")
        
        # Show group-based access
        st.subheader("Access Levels")
        st.write(f"Admin Access: {'✅' if is_user_admin(user) else '❌'}")
        st.write(f"Backup Admin: {'✅' if require_groups(['backup-admin'], user) else '❌'}")
        st.write(f"Readonly Access: {'✅' if require_groups(['readonly', 'user'], user) else '❌'}")
        
        if st.button("Clear Session"):
            from . import clear_session
            clear_session()
            st.success("Session cleared! Refresh the page.")
            
    else:
        st.warning("❌ Not authenticated")
    
    # Development mode authentication setup
    st.subheader("Development Mode Setup")
    st.write("Set test authentication headers:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        username = st.text_input("Username", value="testuser")
        preferred_username = st.text_input("Display Name", value="Test User")
        
    with col2:
        groups_input = st.text_input("Groups (comma-separated)", value="user,backup-admin")
        token = st.text_input("Bearer Token", value="test-token-123")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Set User Headers"):
            groups = [g.strip() for g in groups_input.split(',') if g.strip()]
            set_dev_headers(username, preferred_username, groups, token)
            st.success("Headers set! Refresh to see changes.")
    
    with col2:
        if st.button("Set Admin Headers"):
            set_dev_headers("admin", "Administrator", ["admin", "cluster-admin"], "admin-token")
            st.success("Admin headers set! Refresh to see changes.")
    
    with col3:
        if st.button("Clear Headers"):
            clear_dev_headers()
            st.success("Headers cleared! Refresh to see changes.")


def main():
    """Main function to demonstrate authentication examples."""
    st.sidebar.title("Authentication Examples")
    
    page = st.sidebar.selectbox("Choose Example", [
        "Protected Page",
        "Admin Only Page", 
        "Development Testing"
    ])
    
    if page == "Protected Page":
        protected_page_example()
    elif page == "Admin Only Page":
        admin_only_page()
    elif page == "Development Testing":
        development_testing_page()


if __name__ == "__main__":
    main()