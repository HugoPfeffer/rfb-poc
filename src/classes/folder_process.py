import os
import shutil
from pathlib import Path
from datetime import datetime
import unicodedata
import re
from typing import Dict, List

class FolderProcess:
    def __init__(self):
        """Initialize FolderProcess with project paths."""
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        self.backup_dir = self.project_root / "data" / "backup"
        self.processed_files: Dict[str, str] = {}  # original_name -> standardized_name
        
    def _is_already_processed(self) -> bool:
        """Check if the files in the data directory are already standardized.
        
        Returns:
            bool: True if all files are already in standardized format
        """
        for csv_file in self.data_dir.glob("*.csv"):
            original_name = csv_file.name
            standardized_name = self._normalize_filename(original_name)
            if original_name != standardized_name:
                return False
        return True
        
    def create_backup(self) -> Path:
        """Create a backup of the original CSV files with timestamp.
        
        Returns:
            Path: Path to the created backup directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / timestamp
        
        # Create backup directory if it doesn't exist
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Copy all CSV files to backup directory
        for csv_file in self.data_dir.glob("*.csv"):
            shutil.copy2(csv_file, backup_path / csv_file.name)
            
        print(f"Backup created at: {backup_path}")
        return backup_path
    
    def _normalize_filename(self, filename: str) -> str:
        """Normalize filename by handling special characters and spaces.
        
        Args:
            filename (str): Original filename
            
        Returns:
            str: Normalized filename
        """
        # Remove file extension for processing
        name, ext = os.path.splitext(filename)
        
        # Convert to lowercase
        name = name.lower()
        
        # Normalize special characters (é -> e, ç -> c, etc)
        name = unicodedata.normalize('NFKD', name)
        name = ''.join(c for c in name if not unicodedata.combining(c))
        
        # Replace spaces with underscores and remove any non-alphanumeric chars
        name = re.sub(r'[^a-z0-9]+', '_', name)
        
        # Remove leading/trailing underscores and collapse multiple underscores
        name = re.sub(r'_+', '_', name).strip('_')
        
        return f"{name}{ext}"
    
    def standardize_filenames(self) -> Dict[str, str]:
        """Standardize all CSV filenames in the data directory.
        
        Returns:
            Dict[str, str]: Mapping of original filenames to standardized filenames
        """
        # Check if files are already processed
        if self._is_already_processed():
            print("Files are already in standardized format. Skipping processing.")
            return {}
            
        # First create a backup
        self.create_backup()
        
        # Process each CSV file
        for csv_file in self.data_dir.glob("*.csv"):
            original_name = csv_file.name
            standardized_name = self._normalize_filename(original_name)
            
            if original_name != standardized_name:
                new_path = csv_file.parent / standardized_name
                csv_file.rename(new_path)
                self.processed_files[original_name] = standardized_name
                print(f"Renamed: {original_name} -> {standardized_name}")
            
        return self.processed_files
    
    def get_filename_mapping(self) -> Dict[str, str]:
        """Get the mapping of original to standardized filenames.
        
        Returns:
            Dict[str, str]: Mapping of original filenames to standardized filenames
        """
        return self.processed_files.copy()
    
    def restore_from_backup(self, backup_timestamp: str = None) -> None:
        """Restore files from a specific backup or the most recent one.
        
        Args:
            backup_timestamp (str, optional): Specific backup timestamp to restore from.
                                           If None, uses the most recent backup.
        """
        if not self.backup_dir.exists():
            raise FileNotFoundError("No backup directory found")
            
        # Get all backup directories
        backup_dirs = sorted([d for d in self.backup_dir.iterdir() if d.is_dir()])
        if not backup_dirs:
            raise FileNotFoundError("No backups found")
            
        # Select backup directory
        if backup_timestamp:
            backup_path = self.backup_dir / backup_timestamp
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup {backup_timestamp} not found")
        else:
            backup_path = backup_dirs[-1]  # Most recent backup
            
        # Restore files
        for csv_file in backup_path.glob("*.csv"):
            shutil.copy2(csv_file, self.data_dir / csv_file.name)
            
        print(f"Files restored from backup: {backup_path}")
        
        # Clear the processed files mapping since we restored originals
        self.processed_files.clear() 