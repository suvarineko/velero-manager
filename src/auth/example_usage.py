"""
Example usage of the authentication and session management modules.

This file demonstrates how to integrate the auth module with a Streamlit application.
"""

import streamlit as st
from auth import (
    authenticate_user,
    get_current_user,
    clear_session,
    get_session_manager,
    set_dev_headers
)


def main():
    """Main Streamlit application demonstrating auth integration."""
    
    st.title("Velero Manager - Authentication Example")
    
    # Get session manager instance
    session_manager = get_session_manager()
    
    # Check if user is authenticated
    user = get_current_user()
    
    if user:
        # User is authenticated
        st.success(f"Welcome, {user.preferred_username}!")
        
        # Display user information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("User Information")
            st.write(f"**Username:** {user.username}")
            st.write(f"**Display Name:** {user.preferred_username}")
            st.write(f"**Groups:** {', '.join(user.groups) if user.groups else 'None'}")
            st.write(f"**Has Token:** {'Yes' if user.bearer_token else 'No'}")
        
        with col2:
            st.subheader("Session Information")
            session_info = session_manager.get_session_info()
            st.write(f"**Session ID:** {session_info.get('session_id', 'N/A')[:20]}...")
            st.write(f"**Created:** {session_info.get('created_at', 'N/A')}")
            st.write(f"**Expires:** {session_info.get('expires_at', 'N/A')}")
        
        # Logout button
        if st.button("Logout"):
            clear_session()
            st.rerun()
    
    else:
        # User is not authenticated
        st.warning("You are not authenticated.")
        
        # In production, headers would come from OAuth proxy
        # For development/demo, we can set test headers
        st.subheader("Development Login")
        
        with st.form("dev_login"):
            username = st.text_input("Username", value="john.doe")
            display_name = st.text_input("Display Name", value="John Doe")
            groups = st.text_input("Groups (comma-separated)", value="developers,velero-users")
            token = st.text_input("Bearer Token (optional)", value="test-token-123")
            
            if st.form_submit_button("Set Dev Headers & Login"):
                # Set development headers
                group_list = [g.strip() for g in groups.split(',') if g.strip()]
                set_dev_headers(
                    username=username,
                    preferred_username=display_name,
                    groups=group_list,
                    token=token if token else None
                )
                
                # Attempt authentication
                authenticated_user = authenticate_user()
                if authenticated_user:
                    st.success("Authentication successful!")
                    st.rerun()
                else:
                    st.error("Authentication failed!")
        
        st.info("""
        **Note:** In production, authentication headers would be provided by the OAuth proxy.
        This development login is for testing purposes only.
        """)
    
    # Display raw session state (for debugging)
    with st.expander("Debug: Session State"):
        st.json({
            k: str(v) if not isinstance(v, (dict, list, bool, int, float, type(None))) else v
            for k, v in st.session_state.items()
        })


if __name__ == "__main__":
    main()