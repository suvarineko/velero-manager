"""
Backup UI utilities for Velero Manager.

This module provides functions for creating and tracking backup operations
with real-time progress updates in Streamlit.
"""

import asyncio
import logging
import streamlit as st
from typing import Optional, Dict, Any
from datetime import datetime

try:
    from .backup_manager import BackupManager
    from .auth.auth import UserInfo
except ImportError:
    # Fallback for standalone testing
    from backup_manager import BackupManager
    from auth.auth import UserInfo


logger = logging.getLogger(__name__)


async def create_backup_with_progress(backup_manager: BackupManager, 
                                     namespace: str, 
                                     user: UserInfo,
                                     progress_container: st.container,
                                     status_container: st.container) -> Dict[str, Any]:
    """
    Create a backup with real-time progress tracking in Streamlit UI.
    
    Args:
        backup_manager: BackupManager instance for backup operations
        namespace: Kubernetes namespace to backup
        user: Authenticated user information
        progress_container: Streamlit container for progress updates
        status_container: Streamlit container for status text
        
    Returns:
        Dict containing backup result information:
        - success: bool - whether backup creation succeeded
        - backup_name: str - name of created backup
        - final_status: str - final backup status
        - error: str - error message if failed
        - elapsed_time: float - total time elapsed
    """
    start_time = datetime.now()
    backup_name = None
    result = {
        "success": False,
        "backup_name": None,
        "final_status": "Unknown",
        "error": None,
        "elapsed_time": 0.0
    }
    
    try:
        logger.info(f"Starting backup creation for namespace '{namespace}' by user '{user.username}'")
        
        # Step 1: Generate backup name
        with status_container:
            st.text("🏗️ Generating backup name...")
        
        backup_name = backup_manager.generate_backup_name(namespace, user.username)
        logger.info(f"Generated backup name: {backup_name}")
        
        # Step 2: Initiate backup creation
        with status_container:
            st.text(f"🚀 Creating backup: {backup_name}")
        
        # Store backup info in session state for background monitoring
        if 'active_backups' not in st.session_state:
            st.session_state.active_backups = {}
        
        st.session_state.active_backups[backup_name] = {
            "namespace": namespace,
            "username": user.username,
            "start_time": start_time,
            "status": "Creating"
        }
        
        # Create the backup
        returned_name = await backup_manager.create_backup(
            name=backup_name,
            include_namespaces=[namespace],
            username=user.username
        )
        
        logger.info(f"Backup creation initiated successfully: {returned_name}")
        
        # Step 3: Monitor backup progress
        with status_container:
            st.text(f"📊 Monitoring backup progress...")
        
        final_status = "Unknown"
        async for status in backup_manager.poll_backup_status(backup_name, timeout=1800, poll_interval=5):
            # Update status in session state
            st.session_state.active_backups[backup_name]["status"] = status
            
            # Update UI only on status changes
            with status_container:
                elapsed = (datetime.now() - start_time).total_seconds()
                st.text(f"⏱️ Status: {status} (Elapsed: {elapsed:.0f}s)")
            
            # Log status change
            logger.info(f"Backup '{backup_name}' status: {status}")
            
            # Check for terminal states
            if status.lower() in ["completed", "failed", "partiallyfailed"]:
                final_status = status
                break
        
        # Update final result
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        if final_status.lower() == "completed":
            result.update({
                "success": True,
                "backup_name": backup_name,
                "final_status": final_status,
                "elapsed_time": elapsed_time
            })
            
            with status_container:
                st.success(f"✅ Backup completed successfully in {elapsed_time:.1f}s")
                
        else:
            result.update({
                "success": False,
                "backup_name": backup_name,
                "final_status": final_status,
                "error": f"Backup finished with status: {final_status}",
                "elapsed_time": elapsed_time
            })
            
            with status_container:
                st.error(f"❌ Backup failed with status: {final_status}")
        
        # Clean up from active backups
        if backup_name in st.session_state.active_backups:
            del st.session_state.active_backups[backup_name]
            
    except asyncio.TimeoutError:
        elapsed_time = (datetime.now() - start_time).total_seconds()
        error_msg = f"Backup creation timed out after {elapsed_time:.1f}s"
        
        result.update({
            "success": False,
            "backup_name": backup_name,
            "final_status": "Timeout",
            "error": error_msg,
            "elapsed_time": elapsed_time
        })
        
        with status_container:
            st.error(f"⏰ {error_msg}")
        
        logger.error(f"Backup creation timeout: {error_msg}")
        
    except Exception as e:
        elapsed_time = (datetime.now() - start_time).total_seconds()
        error_msg = f"Backup creation failed: {str(e)}"
        
        result.update({
            "success": False,
            "backup_name": backup_name,
            "final_status": "Error",
            "error": error_msg,
            "elapsed_time": elapsed_time
        })
        
        with status_container:
            st.error(f"💥 {error_msg}")
        
        logger.error(f"Backup creation error: {error_msg}", exc_info=True)
        
        # Clean up from active backups
        if backup_name and backup_name in st.session_state.active_backups:
            del st.session_state.active_backups[backup_name]
    
    logger.info(f"Backup creation finished: {result}")
    return result


def find_running_backups_for_namespace(backup_manager: BackupManager, namespace: str) -> list:
    """
    Find running backups for a specific namespace by querying Velero API.
    
    Args:
        backup_manager: BackupManager instance
        namespace: Namespace to check for running backups
        
    Returns:
        list: List of running backup names for the namespace
    """
    try:
        # Get all backups
        backups = backup_manager.list_backups(include_parsed_metadata=True)
        running_backups = []
        
        for backup in backups:
            # Extract backup metadata
            if 'parsed_metadata' in backup and backup['parsed_metadata']:
                meta = backup['parsed_metadata']
                backup_name = meta.get('name', '')
                backup_status = meta.get('status', '').lower()
                
                # Check if backup is for our namespace and still running
                if (backup_status in ['inprogress', 'pending'] and 
                    backup_name.startswith(f"{namespace}-")):
                    running_backups.append({
                        'name': backup_name,
                        'status': backup_status,
                        'namespace': namespace
                    })
            else:
                # Fallback to raw metadata
                metadata = backup.get('metadata', {})
                status = backup.get('status', {})
                backup_name = metadata.get('name', '')
                backup_status = status.get('phase', '').lower()
                
                if (backup_status in ['inprogress', 'pending'] and 
                    backup_name.startswith(f"{namespace}-")):
                    running_backups.append({
                        'name': backup_name,
                        'status': backup_status,
                        'namespace': namespace
                    })
        
        logger.info(f"Found {len(running_backups)} running backups for namespace '{namespace}'")
        return running_backups
        
    except Exception as e:
        logger.error(f"Error finding running backups for namespace '{namespace}': {e}")
        return []


def check_and_recover_running_backup(backup_manager: BackupManager, namespace: str) -> Optional[dict]:
    """
    Check for running backups in the selected namespace and set up recovery.
    
    Args:
        backup_manager: BackupManager instance  
        namespace: Selected namespace
        
    Returns:
        dict: Running backup info if found, None otherwise
    """
    try:
        running_backups = find_running_backups_for_namespace(backup_manager, namespace)
        
        if running_backups:
            # Take the most recent running backup
            latest_backup = running_backups[0]
            backup_name = latest_backup['name']
            
            # Set up session state for recovery
            st.session_state.backup_in_progress = True
            st.session_state.backup_namespace = namespace
            st.session_state.recovered_backup = {
                'name': backup_name,
                'status': latest_backup['status'],
                'namespace': namespace,
                'recovered': True
            }
            
            logger.info(f"Recovered running backup: {backup_name}")
            return latest_backup
            
        return None
        
    except Exception as e:
        logger.error(f"Error checking for running backups: {e}")
        return None


def display_backup_section_with_progress(namespace_manager, user: UserInfo):
    """
    Display the backup section with progress tracking capabilities.
    
    This function replaces the backup section content during backup creation
    to show real-time progress updates. Now includes automatic detection
    of running backups across page refreshes.
    
    Args:
        namespace_manager: NamespaceManager instance
        user: Authenticated user information
    """
    # Check for ongoing backup operations in session state
    active_backup = None
    if hasattr(st.session_state, 'active_backups') and st.session_state.active_backups:
        # Find most recent active backup
        for backup_name, backup_info in st.session_state.active_backups.items():
            active_backup = (backup_name, backup_info)
            break
    
    # Display background monitoring for active backups
    if active_backup:
        backup_name, backup_info = active_backup
        st.info(f"🔄 Background monitoring: {backup_name}")
        st.caption(f"Status: {backup_info.get('status', 'Unknown')} | "
                  f"Started: {backup_info.get('start_time', 'Unknown')}")
        
        if st.button("View Active Backup", key="view_active"):
            # This could expand to show detailed progress
            st.json(backup_info)
    
    # Check permissions
    from auth.middleware import require_groups
    if not require_groups(["system:authenticated"], user):
        st.warning("You don't have permission to perform backup operations.")
        st.caption("Required groups: admin or backup-admin")
        return
    
    st.info("You have permission to perform backup operations.")
    
    # Auto-detect running backups for selected namespace
    selected_namespace = st.session_state.get('selected_namespace')
    if selected_namespace and namespace_manager:
        # Check if we need to detect running backups
        if not st.session_state.get('backup_in_progress', False):
            from k8s_client import get_k8s_client
            k8s_client = get_k8s_client(user)
            
            if k8s_client and k8s_client.is_authenticated():
                backup_manager_for_check = BackupManager(k8s_client)
                
                # Check for running backups in this namespace
                running_backup = check_and_recover_running_backup(backup_manager_for_check, selected_namespace)
                
                if running_backup:
                    # st.warning(f"🔄 Detected running backup: **{running_backup['name']}**")
                    # st.caption(f"Status: {running_backup['status']} | Click below to monitor progress")
                    
                    # if st.button("Monitor Running Backup", key="monitor_running"):
                    st.rerun()  # This will trigger the progress UI
                    return
    
    # Show resource tree for selected namespace
    if namespace_manager and selected_namespace:
        from main import display_namespace_resources
        display_namespace_resources(namespace_manager, selected_namespace)
    elif not selected_namespace:
        st.info("📁 Please select a namespace to view resources for backup")
        st.caption("Use the namespace dropdown above to select a namespace.")
        return
    
    # Backup creation controls
    selected_namespace = st.session_state.get('selected_namespace')
    
    # Enable backup button if namespace is selected and no active backup
    can_create_backup = (selected_namespace and 
                        namespace_manager and 
                        not active_backup)
    
    if st.button("Create Backup", 
                disabled=not can_create_backup,
                key="create_backup_btn"):
        
        if not selected_namespace:
            st.error("Please select a namespace first")
            return
        
        # Initialize backup creation
        st.session_state.backup_in_progress = True
        st.session_state.backup_namespace = selected_namespace
        st.rerun()


def display_backup_progress_ui(backup_manager: BackupManager, 
                              namespace: str, 
                              user: UserInfo):
    """
    Display the backup progress UI that replaces the backup section content.
    Handles both new backup creation and recovered running backups.
    
    Args:
        backup_manager: BackupManager instance
        namespace: Namespace being backed up
        user: Authenticated user information
    """
    # Check if this is a recovered backup
    recovered_backup = st.session_state.get('recovered_backup')
    
    if recovered_backup and recovered_backup.get('recovered'):
        st.subheader("🔄 Monitoring Recovered Backup")
        backup_name = recovered_backup['name']
        
        # Create containers for dynamic updates
        progress_container = st.container()
        status_container = st.empty()
        
        # Show backup details
        with progress_container:
            st.info(f"Recovered running backup for namespace: **{namespace}**")
            st.info(f"**Backup Name:** {backup_name}")
            st.caption("Backup was already running when page loaded")
        
        # Monitor the recovered backup
        try:
            with status_container:
                st.text("🔍 Checking current backup status...")
            
            # Create new event loop for monitoring
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Create async function to handle monitoring
                async def monitor_backup():
                    final_status = "Unknown"
                    start_time = datetime.now()
                    
                    async for status in backup_manager.poll_backup_status(backup_name, timeout=1800, poll_interval=5):
                        elapsed = (datetime.now() - start_time).total_seconds()
                        
                        with status_container:
                            st.text(f"⏱️ Status: {status} (Monitoring: {elapsed:.0f}s)")
                        
                        if status.lower() in ["completed", "failed", "partiallyfailed"]:
                            final_status = status
                            break
                    
                    return final_status
                
                # Run the monitoring
                final_status = loop.run_until_complete(monitor_backup())
                
                # Show final result
                if final_status.lower() == "completed":
                    st.success(f"✅ Backup completed successfully!")
                    st.info(f"**Backup Name:** {backup_name}")
                else:
                    st.error(f"❌ Backup finished with status: {final_status}")
                    
            finally:
                loop.close()
            
            # Clean up recovered backup state
            if 'recovered_backup' in st.session_state:
                del st.session_state.recovered_backup
                
        except Exception as e:
            st.error(f"💥 Error monitoring recovered backup: {str(e)}")
            logger.error(f"Error monitoring recovered backup: {e}", exc_info=True)
        
        # Back button
        if st.button("Back to Backup Section", key="back_from_recovered"):
            st.session_state.backup_in_progress = False
            if 'backup_namespace' in st.session_state:
                del st.session_state.backup_namespace
            if 'recovered_backup' in st.session_state:
                del st.session_state.recovered_backup
            st.rerun()
            
    else:
        # Normal new backup creation
        st.subheader("💾 Backup in Progress")
        
        # Create containers for dynamic updates
        progress_container = st.container()
        status_container = st.empty()
        
        # Show backup details
        with progress_container:
            st.info(f"Creating backup for namespace: **{namespace}**")
            st.caption(f"Initiated by: {user.preferred_username}")
        
        # Run backup creation with progress tracking
        try:
            # Use asyncio to run the async function
            import asyncio
            
            # Create new event loop for Streamlit
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    create_backup_with_progress(
                        backup_manager, 
                        namespace, 
                        user,
                        progress_container,
                        status_container
                    )
                )
            finally:
                loop.close()
            
            # Show final result
            if result["success"]:
                st.success(f"🎉 Backup created successfully!")
                st.info(f"**Backup Name:** {result['backup_name']}")
                st.info(f"**Duration:** {result['elapsed_time']:.1f} seconds")
            else:
                st.error(f"❌ Backup creation failed")
                if result.get("error"):
                    st.error(f"**Error:** {result['error']}")
            
            # Reset progress state
            if st.button("Back to Backup Section", key="back_to_backup"):
                st.session_state.backup_in_progress = False
                if 'backup_namespace' in st.session_state:
                    del st.session_state.backup_namespace
                st.rerun()
                
        except Exception as e:
            st.error(f"💥 Unexpected error during backup creation: {str(e)}")
            logger.error(f"Error in backup progress UI: {e}", exc_info=True)
            
            # Reset progress state on error
            if st.button("Back to Backup Section", key="back_to_backup_error"):
                st.session_state.backup_in_progress = False
                if 'backup_namespace' in st.session_state:
                    del st.session_state.backup_namespace
                st.rerun()
