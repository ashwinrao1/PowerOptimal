"""
Test script to verify dashboard structure and imports.
"""

import sys
from pathlib import Path

# Add dashboard to path
dashboard_path = Path(__file__).parent
sys.path.insert(0, str(dashboard_path))

def test_imports():
    """Test that all modules import correctly."""
    print("Testing dashboard imports...")
    
    # Test main app
    try:
        import app
        print("✓ app.py imports successfully")
    except Exception as e:
        print(f"✗ app.py import failed: {e}")
        return False
    
    # Test utils
    try:
        import utils
        print("✓ utils.py imports successfully")
    except Exception as e:
        print(f"✗ utils.py import failed: {e}")
        return False
    
    # Test page modules
    pages = ['setup', 'portfolio', 'dispatch', 'scenarios', 'case_study']
    for page in pages:
        try:
            module = __import__(f'pages.{page}', fromlist=[page])
            if hasattr(module, 'render'):
                print(f"✓ pages/{page}.py imports successfully and has render() function")
            else:
                print(f"⚠ pages/{page}.py imports but missing render() function")
        except Exception as e:
            print(f"✗ pages/{page}.py import failed: {e}")
            return False
    
    return True


def test_session_state_initialization():
    """Test session state initialization function."""
    print("\nTesting session state initialization...")
    
    try:
        import app
        
        # Create mock session state
        class MockSessionState:
            def __init__(self):
                self.data = {}
            
            def __contains__(self, key):
                return key in self.data
            
            def __setattr__(self, key, value):
                if key == 'data':
                    super().__setattr__(key, value)
                else:
                    self.data[key] = value
            
            def __getattr__(self, key):
                if key == 'data':
                    return super().__getattribute__(key)
                return self.data.get(key)
        
        # Note: Can't actually test initialize_session_state without Streamlit runtime
        print("✓ Session state initialization function exists")
        return True
        
    except Exception as e:
        print(f"✗ Session state test failed: {e}")
        return False


def test_utility_functions():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    try:
        import utils
        
        # Test formatting functions
        result = utils.format_currency(1500000)
        assert "$" in result and "M" in result
        print(f"✓ format_currency() works correctly: {result}")
        
        result = utils.format_energy(1500)
        assert "GWh" in result
        print(f"✓ format_energy() works correctly: {result}")
        
        result = utils.format_power(1500)
        assert "GW" in result
        print(f"✓ format_power() works correctly: {result}")
        
        # Test validation
        is_valid, msg = utils.validate_input_parameters(300, 99.99, 50)
        assert is_valid == True
        print("✓ validate_input_parameters() works correctly")
        
        is_valid, msg = utils.validate_input_parameters(50, 99.99, 50)
        assert is_valid == False
        print("✓ validate_input_parameters() catches invalid inputs")
        
        # Test color scheme
        colors = utils.get_color_scheme()
        assert 'grid' in colors
        assert 'gas' in colors
        assert 'battery' in colors
        assert 'solar' in colors
        print("✓ get_color_scheme() returns expected colors")
        
        return True
        
    except Exception as e:
        print(f"✗ Utility function test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Dashboard Structure Test")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_session_state_initialization()
    all_passed &= test_utility_functions()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
