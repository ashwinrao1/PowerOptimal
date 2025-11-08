"""
Gurobi WLS configuration loader.

This module loads Gurobi Web License Service (WLS) credentials from a
configuration file and sets the appropriate environment variables.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def load_gurobi_wls_config(config_path: Optional[str] = None) -> bool:
    """
    Load Gurobi WLS credentials from configuration file.
    
    This function reads the WLS credentials from a JSON file and sets the
    required environment variables for Gurobi to use the Web License Service.
    
    Args:
        config_path: Path to the configuration file. If None, uses default
                    location at config/gurobi_wls.json
    
    Returns:
        True if credentials were loaded successfully, False otherwise
    
    Example:
        >>> from src.optimization.gurobi_config import load_gurobi_wls_config
        >>> if load_gurobi_wls_config():
        ...     print("Gurobi WLS configured successfully")
    """
    if config_path is None:
        # Default to config/gurobi_wls.json in project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "gurobi_wls.json"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(
            f"Gurobi WLS config file not found at {config_path}. "
            "Gurobi will use default license configuration."
        )
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        required_fields = ['WLSACCESSID', 'WLSSECRET', 'LICENSEID']
        missing_fields = [field for field in required_fields if not config.get(field)]
        
        if missing_fields:
            logger.warning(
                f"Gurobi WLS config missing required fields: {missing_fields}. "
                "Please fill in config/gurobi_wls.json with your credentials."
            )
            return False
        
        # Set environment variables
        os.environ['WLSACCESSID'] = str(config['WLSACCESSID'])
        os.environ['WLSSECRET'] = str(config['WLSSECRET'])
        os.environ['LICENSEID'] = str(config['LICENSEID'])
        
        logger.info("Gurobi WLS credentials loaded successfully")
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in Gurobi WLS config file: {e}")
        return False
    except Exception as e:
        logger.error(f"Error loading Gurobi WLS config: {e}")
        return False


def get_gurobi_wls_status() -> Dict[str, bool]:
    """
    Check the status of Gurobi WLS configuration.
    
    Returns:
        Dictionary with status information:
        - config_file_exists: Whether the config file exists
        - credentials_set: Whether WLS environment variables are set
        - gurobi_available: Whether gurobipy can be imported
    """
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "config" / "gurobi_wls.json"
    
    status = {
        'config_file_exists': config_path.exists(),
        'credentials_set': all([
            os.environ.get('WLSACCESSID'),
            os.environ.get('WLSSECRET'),
            os.environ.get('LICENSEID')
        ]),
        'gurobi_available': False
    }
    
    try:
        import gurobipy
        status['gurobi_available'] = True
    except ImportError:
        pass
    
    return status


def print_gurobi_setup_instructions():
    """Print instructions for setting up Gurobi WLS credentials."""
    print("="*80)
    print("GUROBI WLS SETUP INSTRUCTIONS")
    print("="*80)
    print("\n1. Open the file: config/gurobi_wls.json")
    print("\n2. Fill in your Gurobi WLS credentials:")
    print("   {")
    print('     "WLSACCESSID": "your-wls-access-id",')
    print('     "WLSSECRET": "your-wls-secret",')
    print('     "LICENSEID": 12345')
    print("   }")
    print("\n3. Save the file")
    print("\n4. Run your optimization script")
    print("\nNote: The config/gurobi_wls.json file is in .gitignore and will not")
    print("      be committed to version control.")
    print("\nTo get WLS credentials:")
    print("  - Go to: https://license.gurobi.com/manager/licenses")
    print("  - Log in with your Gurobi account")
    print("  - Find your WLS license and copy the credentials")
    print("="*80)


if __name__ == "__main__":
    # Test the configuration
    logging.basicConfig(level=logging.INFO)
    
    status = get_gurobi_wls_status()
    
    print("\nGurobi WLS Configuration Status:")
    print(f"  Config file exists: {status['config_file_exists']}")
    print(f"  Credentials set: {status['credentials_set']}")
    print(f"  Gurobi available: {status['gurobi_available']}")
    
    if not status['config_file_exists']:
        print("\n⚠️  Config file not found!")
        print_gurobi_setup_instructions()
    elif not status['credentials_set']:
        print("\n⚠️  Credentials not configured!")
        print_gurobi_setup_instructions()
    else:
        print("\n✓ Gurobi WLS is configured!")
        
        if status['gurobi_available']:
            print("✓ Gurobi Python package is installed!")
            
            # Try to create a test model
            try:
                import gurobipy as gp
                load_gurobi_wls_config()
                m = gp.Model()
                print(f"✓ Gurobi license is valid! (Version {gp.gurobi.version()})")
            except Exception as e:
                print(f"✗ Error testing Gurobi license: {e}")
        else:
            print("⚠️  Gurobi Python package not installed")
            print("    Install with: pip install gurobipy")
