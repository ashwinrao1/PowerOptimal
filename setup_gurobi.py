#!/usr/bin/env python
"""
Interactive setup script for Gurobi WLS credentials.

This script helps you configure your Gurobi Web License Service credentials
in a secure configuration file.
"""

import json
from pathlib import Path


def main():
    print("="*80)
    print("GUROBI WLS SETUP")
    print("="*80)
    print("\nThis script will help you configure your Gurobi WLS credentials.")
    print("\nYou can find your credentials at:")
    print("  https://license.gurobi.com/manager/licenses")
    print("\n" + "="*80)
    
    # Get credentials from user
    print("\nEnter your Gurobi WLS credentials:")
    wls_access_id = input("WLSACCESSID: ").strip()
    wls_secret = input("WLSSECRET: ").strip()
    license_id = input("LICENSEID: ").strip()
    
    # Validate inputs
    if not wls_access_id or not wls_secret or not license_id:
        print("\n❌ Error: All fields are required!")
        return 1
    
    try:
        license_id = int(license_id)
    except ValueError:
        print("\n❌ Error: LICENSEID must be a number!")
        return 1
    
    # Create config
    config = {
        "WLSACCESSID": wls_access_id,
        "WLSSECRET": wls_secret,
        "LICENSEID": license_id
    }
    
    # Save to file
    config_path = Path("config/gurobi_wls.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*80)
    print("✓ Configuration saved to config/gurobi_wls.json")
    print("="*80)
    
    # Test the configuration
    print("\nTesting configuration...")
    
    try:
        from src.optimization.gurobi_config import load_gurobi_wls_config, get_gurobi_wls_status
        
        load_gurobi_wls_config()
        status = get_gurobi_wls_status()
        
        print(f"  Config file exists: {'✓' if status['config_file_exists'] else '✗'}")
        print(f"  Credentials set: {'✓' if status['credentials_set'] else '✗'}")
        print(f"  Gurobi available: {'✓' if status['gurobi_available'] else '✗'}")
        
        if status['gurobi_available']:
            try:
                import gurobipy as gp
                m = gp.Model()
                print(f"\n✓ Gurobi license is valid! (Version {gp.gurobi.version()})")
                print("\nYou're all set! You can now run optimization scripts with Gurobi.")
            except Exception as e:
                print(f"\n⚠️  Warning: Could not validate license: {e}")
                print("\nThe configuration is saved, but there may be an issue with your license.")
                print("Please verify your credentials at: https://license.gurobi.com/manager/licenses")
        else:
            print("\n⚠️  Gurobi Python package is not installed.")
            print("Install it with: pip install gurobipy")
    
    except Exception as e:
        print(f"\n⚠️  Could not test configuration: {e}")
        print("The configuration is saved, but you may need to verify it manually.")
    
    print("\n" + "="*80)
    return 0


if __name__ == "__main__":
    exit(main())
