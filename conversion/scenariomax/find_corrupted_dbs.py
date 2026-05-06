"""
Find and move corrupted SQLite database files to a separate folder.
"""
import sqlite3
import shutil
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_db_integrity(db_path: Path) -> bool:
    """
    Check if a SQLite database file is corrupted.
    
    Args:
        db_path: Path to the database file
        
    Returns:
        True if database is valid, False if corrupted
    """
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        result = cursor.fetchone()
        conn.close()
        return result[0] == "ok"
    except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
        logger.debug(f"Database error for {db_path}: {e}")
        return False
    except Exception as e:
        logger.warning(f"Unexpected error checking {db_path}: {e}")
        return False

def find_and_move_corrupted_dbs(data_dir: str, corrupted_dir: str = "corrupted"):
    """
    Find all corrupted .db files and move them to a separate directory.
    
    Args:
        data_dir: Root directory to search for .db files
        corrupted_dir: Directory name to move corrupted files (relative to data_dir)
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return
    
    # Create corrupted directory at the same level as data_dir
    corrupted_path = data_path.parent / corrupted_dir
    corrupted_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Corrupted files will be moved to: {corrupted_path}")
    
    # Find all .db files
    db_files = list(data_path.rglob("*.db"))
    logger.info(f"🔍 Found {len(db_files)} database files to check")
    
    corrupted_count = 0
    valid_count = 0
    
    for db_file in db_files:
        logger.info(f"Checking: {db_file.relative_to(data_path)}")
        
        if not check_db_integrity(db_file):
            corrupted_count += 1
            logger.warning(f"❌ CORRUPTED: {db_file.name}")
            
            # Create subdirectory structure in corrupted folder
            relative_path = db_file.relative_to(data_path)
            dest_path = corrupted_path / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move the file
            try:
                shutil.move(str(db_file), str(dest_path))
                logger.info(f"  ➡️  Moved to: {dest_path.relative_to(corrupted_path.parent)}")
            except Exception as e:
                logger.error(f"  ⚠️  Failed to move file: {e}")
        else:
            valid_count += 1
            logger.info(f"✅ Valid: {db_file.name}")
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"📊 Summary:")
    logger.info(f"  Total files checked: {len(db_files)}")
    logger.info(f"  Valid databases: {valid_count}")
    logger.info(f"  Corrupted databases: {corrupted_count}")
    logger.info(f"  Corrupted files location: {corrupted_path}")
    logger.info("=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Find and move corrupted SQLite database files"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/data/cache/train_vegas_2",
        help="Directory to search for database files"
    )
    parser.add_argument(
        "--corrupted_dir",
        type=str,
        default="corrupted",
        help="Directory name for corrupted files (created in parent of data_dir)"
    )
    
    args = parser.parse_args()
    
    find_and_move_corrupted_dbs(args.data_dir, args.corrupted_dir)