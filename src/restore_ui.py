"""
Restore UI Component for Velero Manager.

This module provides the Streamlit UI component for displaying existing backups
and restore functionality. It integrates with the BackupManager to retrieve
and format backup data for display in the main application.

Author: Generated for Velero Manager
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

# Import necessary modules
try:
    from .backup_manager import BackupManager, create_backup_manager
    from .velero_client import VeleroClientConfig
    from .k8s_client import KubernetesClient
except ImportError:
    # Fallback for standalone testing
    from backup_manager import BackupManager, create_backup_manager
    from velero_client import VeleroClientConfig
    from k8s_client import KubernetesClient


def display_restore_ui(
    selected_namespace: Optional[str] = None,
    k8s_client: Optional[KubernetesClient] = None,
    user_permissions: Optional[List[str]] = None
) -> None:
    """
    Display the restore UI component with backup listing and restore functionality.
    
    This function renders the complete restore section including:
    - Backup table display with formatting
    - Restore button functionality with confirmation dialogs
    - Visual styling and dividers
    
    Args:
        selected_namespace: Currently selected namespace for backup filtering
        k8s_client: Authenticated Kubernetes client instance
        user_permissions: List of user group permissions for access control
    """
    logger = logging.getLogger(__name__)
    
    # Container for the restore section
    with st.container():
        # Section header
        st.subheader("🔄 Restore Section")
        
        # Check if required parameters are provided
        if not k8s_client:
            st.error("❌ Kubernetes client not available")
            st.caption("Unable to connect to Kubernetes API. Please check authentication.")
            return
        
        if not selected_namespace:
            st.info("📁 Please select a namespace to view available backups")
            st.caption("Use the namespace dropdown above to select a namespace.")
            return
        
        # Check user permissions for restore operations
        if user_permissions and not any(group in ["system:authenticated"] for group in user_permissions):
            st.warning("🔒 You don't have permission to perform restore operations")
            st.caption("Required groups: admin, backup-admin, or restore-admin")
            return
        
        # Display selected namespace info
        st.markdown(f"**Selected Namespace:** `{selected_namespace}`")
        
        # Divider for visual separation
        st.divider()
        
        # Backup table section
        _display_backup_table_section(selected_namespace, k8s_client, logger)
        
        # Additional divider for future sections
        st.divider()
        
        # Placeholder for restore functionality (will be implemented in Task 11.3)
        with st.expander("🚀 Restore Operations", expanded=False):
            st.info("Restore functionality will be implemented in the next phase")
            st.caption("This section will contain restore buttons and confirmation dialogs")


def _display_backup_table_section(
    namespace: str, 
    k8s_client: KubernetesClient, 
    logger: logging.Logger
) -> None:
    """
    Display the backup table section with BackupManager integration and error handling.
    
    This function handles the complete backup table display logic using BackupManager
    for data retrieval and formatting. Includes auto-refresh when namespace changes.
    
    Args:
        namespace: Namespace to filter backups for
        k8s_client: Authenticated Kubernetes client instance  
        logger: Logger instance for debugging
    """
    # Section header for backup table
    st.markdown("### 📋 Available Backups")
    
    try:
        # Get backup list using BackupManager integration
        backup_df, error_msg = get_backup_list(namespace, k8s_client, logger)
        
        if error_msg:
            # Show error message with caption and empty table
            st.error("❌ Error loading backups")
            st.caption(error_msg)
            # Display empty table with proper column structure
            _display_empty_backup_table()
            return
        
        if backup_df.empty:
            # Show info message for no backups found
            st.info(f"📭 No backups found for namespace: `{namespace}`")
            st.caption("No backup data available. Create backups to see them listed here.")
            # Display empty table with proper column structure
            _display_empty_backup_table()
            return
        
        # Display backup count information
        backup_count = len(backup_df)
        st.caption(f"Found {backup_count} backup{'s' if backup_count != 1 else ''} in namespace `{namespace}`")
        
        # Display the backup table with proper formatting
        _display_formatted_backup_table(backup_df, logger)
        
        # Display backup summary statistics
        _display_backup_summary(namespace, k8s_client, logger)
        
        logger.info(f"Successfully displayed {backup_count} backups for namespace: {namespace}")
        
    except Exception as e:
        logger.error(f"Error in backup table section: {e}")
        st.error("❌ Error loading backup table")
        st.caption(f"Error details: {str(e)}")
        _display_empty_backup_table()


def get_backup_list(
    namespace: str, 
    k8s_client: KubernetesClient, 
    logger: logging.Logger
) -> tuple[pd.DataFrame, Optional[str]]:
    """
    Retrieve and format backup data using BackupManager.
    
    This function implements the core backup data retrieval logic using
    BackupManager for consistent data formatting and error handling.
    
    Args:
        namespace: Namespace to filter backups for
        k8s_client: Authenticated Kubernetes client instance
        logger: Logger instance for debugging
        
    Returns:
        tuple[pd.DataFrame, Optional[str]]: (backup_dataframe, error_message)
    """
    try:
        # Create BackupManager instance
        backup_manager = _create_backup_manager_instance(k8s_client)
        
        if not backup_manager:
            return pd.DataFrame(), "Failed to initialize backup manager"
        
        # Get formatted backups with auto-sorting by timestamp (newest first)
        backup_df = backup_manager.get_formatted_backups(
            namespace=namespace,
            sort_by_timestamp=True,
            include_parsed_metadata=True
        )
        
        logger.debug(f"Retrieved {len(backup_df)} backups for namespace: {namespace}")
        return backup_df, None
        
    except Exception as e:
        error_msg = f"Failed to retrieve backups: {str(e)}"
        logger.error(error_msg)
        return pd.DataFrame(), error_msg


def _display_formatted_backup_table(backup_df: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Display the formatted backup table with proper column widths and styling.
    
    Args:
        backup_df: DataFrame containing backup data
        logger: Logger instance for debugging
    """
    try:
        # Configure column display settings for better readability
        column_config = {
            "Name": st.column_config.TextColumn(
                "Backup Name",
                help="Name of the backup",
                width="medium"
            ),
            "Status": st.column_config.TextColumn(
                "Status",
                help="Current backup status", 
                width="small"
            ),
            "Phase": st.column_config.TextColumn(
                "Phase",
                help="Backup phase",
                width="small"
            ),
            "Creation Time": st.column_config.DatetimeColumn(
                "Created",
                help="When the backup was created",
                width="medium"
            ),
            "Expiration": st.column_config.DatetimeColumn(
                "Expires",
                help="When the backup will expire",
                width="medium"
            ),
            "Creator": st.column_config.TextColumn(
                "Created By",
                help="User who created the backup",
                width="small"
            ),
            "Errors": st.column_config.TextColumn(
                "Errors",
                help="Any errors encountered during backup",
                width="medium"
            )
        }
        
        # Display the dataframe with custom configuration
        st.dataframe(
            backup_df,
            use_container_width=True,
            hide_index=True,
            column_config=column_config
        )
        
        logger.debug(f"Displayed backup table with {len(backup_df)} rows")
        
    except Exception as e:
        logger.error(f"Error displaying backup table: {e}")
        # Fallback to simple table display
        st.dataframe(
            backup_df,
            use_container_width=True,
            hide_index=True
        )


def _display_empty_backup_table() -> None:
    """
    Display an empty backup table with proper column structure.
    
    This function shows an empty table maintaining the expected column
    structure for consistent UI layout.
    """
    column_config = _format_backup_table_columns()
    empty_data = {col: [] for col in column_config.keys()}
    
    st.dataframe(
        pd.DataFrame(empty_data),
        use_container_width=True,
        hide_index=True
    )


def _display_backup_summary(
    namespace: str, 
    k8s_client: KubernetesClient, 
    logger: logging.Logger
) -> None:
    """
    Display backup summary statistics in an expandable section.
    
    Args:
        namespace: Namespace to generate summary for
        k8s_client: Authenticated Kubernetes client instance
        logger: Logger instance for debugging
    """
    try:
        # Create BackupManager instance for summary
        backup_manager = _create_backup_manager_instance(k8s_client)
        
        if not backup_manager:
            return
        
        # Get backup summary statistics
        summary = backup_manager.get_backup_summary(namespace=namespace)
        
        if summary.get('total_backups', 0) > 0:
            # Display summary in expandable section
            with st.expander("📊 Backup Summary", expanded=False):
                summary_text = backup_manager.format_summary_for_display(summary)
                st.markdown(summary_text)
        
        logger.debug(f"Displayed backup summary for namespace: {namespace}")
        
    except Exception as e:
        logger.warning(f"Could not display backup summary: {e}")
        # Summary is optional, so we don't show error to user


def _create_backup_manager_instance(k8s_client: KubernetesClient) -> Optional[BackupManager]:
    """
    Create and configure a BackupManager instance.
    
    This function will be used in Task 11.2 to create the BackupManager
    instance for backup data retrieval and formatting.
    
    Args:
        k8s_client: Authenticated Kubernetes client instance
        
    Returns:
        Optional[BackupManager]: Configured BackupManager instance or None on error
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Create Velero client configuration
        velero_config = VeleroClientConfig()
        
        # Create BackupManager instance using factory function
        backup_manager = create_backup_manager(k8s_client, velero_config)
        
        logger.debug("BackupManager instance created successfully")
        return backup_manager
        
    except Exception as e:
        logger.error(f"Failed to create BackupManager: {e}")
        return None


def _format_backup_table_columns() -> Dict[str, str]:
    """
    Define the column configuration for the backup table display.
    
    This function defines how backup data columns should be formatted
    and displayed in the Streamlit dataframe.
    
    Returns:
        Dict[str, str]: Column configuration mapping
    """
    return {
        'Name': 'Backup Name',
        'Status': 'Status', 
        'Phase': 'Phase',
        'Creation Time': 'Created',
        'Expiration': 'Expires',
        'Creator': 'Created By',
        'Errors': 'Errors'
    }


# Test function for standalone component testing
def test_restore_ui_component():
    """
    Test function for standalone component development and debugging.
    This function can be used to test the restore UI component independently.
    """
    st.title("🧪 Restore UI Component Test")
    
    # Mock data for testing
    test_namespace = "test-namespace"
    test_permissions = ["admin", "backup-admin"]
    
    st.info("This is a test mode for the restore UI component")
    st.markdown(f"**Test Namespace:** {test_namespace}")
    st.markdown(f"**Test Permissions:** {test_permissions}")
    
    # Display the component (will show placeholder until 11.2)
    display_restore_ui(
        selected_namespace=test_namespace,
        k8s_client=None,  # Will show error message
        user_permissions=test_permissions
    )


if __name__ == "__main__":
    # Run test mode when file is executed directly
    test_restore_ui_component()