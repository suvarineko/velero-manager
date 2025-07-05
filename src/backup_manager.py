"""
BackupManager module for Velero Manager.

This module provides the BackupManager class for formatting and presenting 
backup data in Streamlit-friendly formats. It extends VeleroClient to provide
enhanced data presentation capabilities specifically for the Streamlit UI.

Author: Generated for Velero Manager
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, AsyncIterator
from datetime import datetime, timezone, timedelta
import pandas as pd
import re

try:
    from .velero_client import VeleroClient, VeleroClientConfig
except ImportError:
    # Fallback for standalone testing
    from velero_client import VeleroClient, VeleroClientConfig


class BackupManager(VeleroClient):
    """
    BackupManager extends VeleroClient to provide enhanced backup data formatting 
    and presentation capabilities for Streamlit UI.
    
    This class inherits all VeleroClient functionality while adding methods for:
    - Formatting backup data for Streamlit tables
    - Simple timestamp-based sorting
    - Summary statistics generation
    - UI-friendly data structure conversion
    """
    
    def __init__(self, k8s_client, config: Optional[VeleroClientConfig] = None):
        """
        Initialize BackupManager with KubernetesClient and optional configuration.
        
        Args:
            k8s_client: Authenticated KubernetesClient instance
            config: Optional VeleroClientConfig (defaults will be used if None)
        """
        super().__init__(k8s_client, config)
        self.logger = logging.getLogger(__name__)
        self.logger.debug("BackupManager initialized")
    
    def generate_backup_name(self, namespace: str, username: str) -> str:
        """
        Generate a standardized backup name based on namespace, timestamp, and username.
        
        Format: {namespace}-{timestamp}-{username}
        Timestamp format: YYYY-MM-DD-HHMMSS (UTC+3 timezone)
        
        Args:
            namespace: Kubernetes namespace name
            username: Username of the user triggering the backup creation
            
        Returns:
            str: Generated backup name that meets Velero naming requirements
            
        Raises:
            ValueError: If generated name doesn't meet Velero naming requirements
        """
        try:
            # Get current timestamp in UTC+3 timezone
            utc_plus_3 = timezone(timedelta(hours=3))
            now = datetime.now(utc_plus_3)
            timestamp = now.strftime('%Y-%m-%d-%H%M%S')
            
            # Sanitize inputs (convert to lowercase)
            namespace_clean = namespace.lower()
            username_clean = username.lower()
            
            # Generate backup name
            backup_name = f"{namespace_clean}-{timestamp}-{username_clean}"
            
            # Validate the generated name meets Velero requirements
            self._validate_backup_name(backup_name)
            
            self.logger.debug(f"Generated backup name: {backup_name}")
            return backup_name
            
        except Exception as e:
            error_msg = f"Failed to generate backup name for namespace '{namespace}' and user '{username}': {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg) from e
    
    def _validate_backup_name(self, backup_name: str) -> None:
        """
        Validate that the backup name meets Velero's naming requirements.
        
        Velero backup names must:
        - Be lowercase
        - Start and end with alphanumeric characters
        - Contain only lowercase alphanumeric characters, hyphens, and dots
        - Be 253 characters or less
        
        Args:
            backup_name: The backup name to validate
            
        Raises:
            ValueError: If the backup name doesn't meet requirements
        """
        try:
            # Check length requirement (253 characters max)
            if len(backup_name) > 253:
                raise ValueError(f"Backup name exceeds 253 character limit: {len(backup_name)} characters")
            
            # Check if name is empty
            if not backup_name:
                raise ValueError("Backup name cannot be empty")
            
            # Check start and end characters (must be alphanumeric)
            if not (backup_name[0].isalnum() and backup_name[-1].isalnum()):
                raise ValueError("Backup name must start and end with alphanumeric characters")
            
            # Check allowed characters (lowercase alphanumeric, hyphens, dots)
            # Velero follows Kubernetes naming conventions
            valid_pattern = re.compile(r'^[a-z0-9]([a-z0-9\-\.]*[a-z0-9])?$')
            if not valid_pattern.match(backup_name):
                raise ValueError("Backup name contains invalid characters. Only lowercase letters, numbers, hyphens, and dots are allowed")
            
            # Additional check: ensure no consecutive dots or hyphens at start/end
            if '..' in backup_name:
                raise ValueError("Backup name cannot contain consecutive dots")
            
            # Check for valid DNS subdomain name (Kubernetes requirement)
            if backup_name.startswith('-') or backup_name.endswith('-'):
                raise ValueError("Backup name cannot start or end with a hyphen")
            
            if backup_name.startswith('.') or backup_name.endswith('.'):
                raise ValueError("Backup name cannot start or end with a dot")
            
            self.logger.debug(f"Backup name validation passed: {backup_name}")
            
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Unexpected error during backup name validation: {e}") from e
    
    def get_formatted_backups(
        self, 
        namespace: Optional[str] = None,
        sort_by_timestamp: bool = True,
        include_parsed_metadata: bool = True
    ) -> pd.DataFrame:
        """
        Get backups formatted as a Streamlit-ready DataFrame.
        
        Args:
            namespace: Optional namespace filter
            sort_by_timestamp: Sort by creation timestamp (newest first)
            include_parsed_metadata: Include parsed metadata in results
            
        Returns:
            pandas.DataFrame: Formatted backup data ready for st.dataframe()
        """
        try:
            # Get backups using inherited VeleroClient functionality
            backups = self.list_backups(
                namespace=namespace, 
                include_parsed_metadata=include_parsed_metadata
            )
            
            if not backups:
                self.logger.info(f"No backups found for namespace: {namespace}")
                return pd.DataFrame()  # Return empty DataFrame
            
            # Convert to DataFrame format
            formatted_data = []
            for backup in backups:
                row = self._format_backup_row(backup, include_parsed_metadata)
                formatted_data.append(row)
            
            df = pd.DataFrame(formatted_data)
            
            # Apply initial sorting by timestamp if requested
            if sort_by_timestamp and not df.empty and 'Creation Time' in df.columns:
                df = df.sort_values('Creation Time', ascending=False).reset_index(drop=True)
                self.logger.debug(f"Sorted {len(df)} backups by timestamp (newest first)")
            
            self.logger.info(f"Formatted {len(df)} backups for display")
            return df
            
        except Exception as e:
            self.logger.error(f"Error formatting backups: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def _format_backup_row(self, backup: Dict[str, Any], include_parsed: bool = True) -> Dict[str, str]:
        """
        Format a single backup entry for DataFrame display.
        
        Args:
            backup: Raw backup data from VeleroClient
            include_parsed: Whether to use parsed metadata
            
        Returns:
            Dict[str, str]: Formatted row data for DataFrame
        """
        try:
            # Use parsed metadata if available and requested
            if include_parsed and 'parsed_metadata' in backup and backup['parsed_metadata']:
                meta = backup['parsed_metadata']
                
                # Format creation timestamp
                creation_time = meta.get('creation_timestamp')
                if isinstance(creation_time, datetime):
                    creation_str = creation_time.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    creation_str = str(creation_time) if creation_time else 'Unknown'
                
                # Format expiration date
                expiration = meta.get('expiration_date')
                if isinstance(expiration, datetime):
                    expiration_str = expiration.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    expiration_str = str(expiration) if expiration else 'Unknown'
                
                return {
                    'Name': meta.get('name', 'Unknown'),
                    'Status': meta.get('status', 'Unknown'),
                    'Phase': meta.get('phase', 'Unknown'),
                    'Creation Time': creation_str,
                    'Expiration': expiration_str,
                    'Creator': meta.get('creator', 'Unknown'),
                    'Errors': ', '.join(meta.get('errors', [])) or 'None'
                }
            
            else:
                # Fallback to raw metadata
                metadata = backup.get('metadata', {})
                status = backup.get('status', {})
                
                return {
                    'Name': metadata.get('name', 'Unknown'),
                    'Status': status.get('phase', 'Unknown'),
                    'Phase': status.get('phase', 'Unknown'),
                    'Creation Time': metadata.get('creationTimestamp', 'Unknown'),
                    'Expiration': status.get('expiration', 'Unknown'),
                    'Creator': 'Unknown',  # Raw data doesn't have parsed creator
                    'Errors': 'Unknown'    # Raw data doesn't have parsed errors
                }
                
        except Exception as e:
            self.logger.warning(f"Error formatting backup row: {e}")
            # Return safe fallback row
            return {
                'Name': str(backup.get('metadata', {}).get('name', 'Error')),
                'Status': 'Error',
                'Phase': 'Error', 
                'Creation Time': 'Error',
                'Expiration': 'Error',
                'Creator': 'Error',
                'Errors': str(e)
            }
    
    def get_backup_summary(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate summary statistics for backups.
        
        Args:
            namespace: Optional namespace filter
            
        Returns:
            Dict[str, Any]: Summary statistics including counts and status distribution
        """
        try:
            # Get backups with parsed metadata
            backups = self.list_backups(namespace=namespace, include_parsed_metadata=True)
            
            if not backups:
                return {
                    'total_backups': 0,
                    'namespace': namespace or 'All namespaces',
                    'status_distribution': {},
                    'phase_distribution': {},
                    'oldest_backup': None,
                    'newest_backup': None,
                    'creators': []
                }
            
            # Count statistics
            total_backups = len(backups)
            status_counts = {}
            phase_counts = {}
            creators = set()
            timestamps = []
            
            for backup in backups:
                # Extract status and phase information
                if 'parsed_metadata' in backup and backup['parsed_metadata']:
                    meta = backup['parsed_metadata']
                    status = meta.get('status', 'Unknown')
                    phase = meta.get('phase', 'Unknown')
                    creator = meta.get('creator', 'Unknown')
                    timestamp = meta.get('creation_timestamp')
                else:
                    # Fallback to raw data
                    status = backup.get('status', {}).get('phase', 'Unknown')
                    phase = status
                    creator = 'Unknown'
                    timestamp = backup.get('metadata', {}).get('creationTimestamp')
                
                # Count distributions
                status_counts[status] = status_counts.get(status, 0) + 1
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
                
                if creator != 'Unknown':
                    creators.add(creator)
                
                if isinstance(timestamp, datetime):
                    timestamps.append(timestamp)
            
            # Calculate date range
            oldest_backup = None
            newest_backup = None
            if timestamps:
                oldest_backup = min(timestamps).strftime('%Y-%m-%d %H:%M:%S')
                newest_backup = max(timestamps).strftime('%Y-%m-%d %H:%M:%S')
            
            summary = {
                'total_backups': total_backups,
                'namespace': namespace or 'All namespaces',
                'status_distribution': status_counts,
                'phase_distribution': phase_counts,
                'oldest_backup': oldest_backup,
                'newest_backup': newest_backup,
                'creators': list(creators)
            }
            
            self.logger.info(f"Generated summary for {total_backups} backups in namespace: {namespace}")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating backup summary: {e}")
            return {
                'total_backups': 0,
                'namespace': namespace or 'All namespaces',
                'status_distribution': {'Error': 1},
                'phase_distribution': {'Error': 1},
                'oldest_backup': None,
                'newest_backup': None,
                'creators': [],
                'error': str(e)
            }
    
    def get_backup_by_name(self, backup_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific backup by name.
        
        Args:
            backup_name: Name of the backup to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Backup details or None if not found
        """
        try:
            # Get all backups with parsed metadata
            backups = self.list_backups(include_parsed_metadata=True)
            
            for backup in backups:
                # Check both raw and parsed metadata for name match
                backup_found = False
                
                if 'parsed_metadata' in backup and backup['parsed_metadata']:
                    if backup['parsed_metadata'].get('name') == backup_name:
                        backup_found = True
                elif backup.get('metadata', {}).get('name') == backup_name:
                    backup_found = True
                
                if backup_found:
                    self.logger.debug(f"Found backup: {backup_name}")
                    return backup
            
            self.logger.warning(f"Backup not found: {backup_name}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving backup {backup_name}: {e}")
            return None
    
    def format_summary_for_display(self, summary: Dict[str, Any]) -> str:
        """
        Format backup summary for Streamlit display.
        
        Args:
            summary: Summary data from get_backup_summary()
            
        Returns:
            str: Formatted summary text for display
        """
        try:
            lines = []
            lines.append(f"**Backup Summary for {summary['namespace']}**")
            lines.append(f"Total backups: {summary['total_backups']}")
            
            if summary['total_backups'] > 0:
                # Status distribution
                if summary['status_distribution']:
                    lines.append("\n**Status Distribution:**")
                    for status, count in summary['status_distribution'].items():
                        lines.append(f"- {status}: {count}")
                
                # Date range
                if summary['oldest_backup'] and summary['newest_backup']:
                    lines.append(f"\n**Date Range:**")
                    lines.append(f"- Oldest: {summary['oldest_backup']}")
                    lines.append(f"- Newest: {summary['newest_backup']}")
                
                # Creators
                if summary['creators']:
                    lines.append(f"\n**Creators:** {', '.join(summary['creators'])}")
            
            if 'error' in summary:
                lines.append(f"\n**Error:** {summary['error']}")
            
            return '\n'.join(lines)
            
        except Exception as e:
            self.logger.error(f"Error formatting summary: {e}")
            return f"Error formatting summary: {e}"
    
    async def poll_backup_status(self, backup_name: str, 
                                timeout: int = 1800, 
                                poll_interval: int = 5) -> AsyncIterator[str]:
        """
        Poll backup status asynchronously until completion or timeout.
        
        Args:
            backup_name: Name of the backup to monitor
            timeout: Maximum time to poll in seconds (default: 1800s = 30 minutes)
            poll_interval: Interval between polls in seconds (default: 5 seconds)
            
        Yields:
            str: Current backup status ("Pending", "InProgress", "Completed", "Failed", etc.)
            
        Raises:
            asyncio.TimeoutError: If polling exceeds timeout duration
            Exception: If backup status checking fails repeatedly
        """
        self.logger.info(f"Starting status polling for backup: {backup_name}")
        self.logger.debug(f"Polling configuration - timeout: {timeout}s, interval: {poll_interval}s")
        
        start_time = datetime.now()
        last_status = None
        error_count = 0
        max_consecutive_errors = 5
        
        try:
            while True:
                # Check timeout
                elapsed_time = (datetime.now() - start_time).total_seconds()
                if elapsed_time >= timeout:
                    self.logger.warning(f"Polling timeout reached for backup '{backup_name}' after {elapsed_time:.1f}s")
                    raise asyncio.TimeoutError(f"Backup status polling timed out after {timeout} seconds")
                
                try:
                    # Get current backup status
                    current_status = self.get_backup_status(backup_name, raise_on_error=False)
                    
                    # Reset error count on successful status check
                    error_count = 0
                    
                    # Log status changes
                    if current_status != last_status:
                        self.logger.info(f"Backup '{backup_name}' status changed: {last_status} -> {current_status}")
                        last_status = current_status
                    
                    # Yield current status
                    yield current_status
                    
                    # Check for terminal states
                    if current_status.lower() in ["completed", "failed", "partiallyfailed"]:
                        self.logger.info(f"Backup '{backup_name}' reached terminal state: {current_status}")
                        break
                    
                    # Handle error states
                    if current_status.lower() in ["error", "notfound"]:
                        self.logger.warning(f"Backup '{backup_name}' in error state: {current_status}")
                        # Continue polling for a few more iterations in case it's transient
                        error_count += 1
                        if error_count >= 3:
                            self.logger.error(f"Backup '{backup_name}' persistently in error state: {current_status}")
                            break
                    
                except Exception as e:
                    error_count += 1
                    self.logger.warning(f"Error checking backup status (attempt {error_count}/{max_consecutive_errors}): {e}")
                    
                    if error_count >= max_consecutive_errors:
                        self.logger.error(f"Too many consecutive errors checking backup status for '{backup_name}'")
                        raise Exception(f"Failed to check backup status after {max_consecutive_errors} attempts: {e}")
                    
                    # Yield error status and continue
                    yield "Error"
                
                # Wait before next poll
                await asyncio.sleep(poll_interval)
                
        except asyncio.TimeoutError:
            # Re-raise timeout errors
            raise
        except Exception as e:
            self.logger.error(f"Polling failed for backup '{backup_name}': {e}")
            raise
        finally:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Finished polling backup '{backup_name}' after {elapsed_time:.1f}s")


def create_backup_manager(k8s_client, config: Optional[VeleroClientConfig] = None) -> BackupManager:
    """
    Factory function to create a BackupManager instance.
    
    Args:
        k8s_client: Authenticated KubernetesClient instance
        config: Optional VeleroClientConfig
        
    Returns:
        BackupManager: Configured BackupManager instance
    """
    return BackupManager(k8s_client, config)