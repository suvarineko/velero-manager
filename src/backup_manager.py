"""
BackupManager module for Velero Manager.

This module provides the BackupManager class for formatting and presenting 
backup data in Streamlit-friendly formats. It extends VeleroClient to provide
enhanced data presentation capabilities specifically for the Streamlit UI.

Author: Generated for Velero Manager
"""

import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import pandas as pd

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