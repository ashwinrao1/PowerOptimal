"""
Simple test to verify scenarios page can be imported and basic functions work.
"""

import sys
from pathlib import Path

# Add paths
dashboard_path = Path(__file__).parent
src_path = dashboard_path.parent / "src"
sys.path.insert(0, str(dashboard_path))
sys.path.insert(0, str(src_path))

def test_import():
    """Test that scenarios page can be imported."""
    try:
        from pages import scenarios
        print("✓ Scenarios page imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import scenarios page: {e}")
        return False


def test_scenario_config_structure():
    """Test scenario configuration structure."""
    try:
        from pages import scenarios
        
        # Test that key functions exist
        assert hasattr(scenarios, 'render'), "render function not found"
        assert hasattr(scenarios, 'render_scenario_configuration'), "render_scenario_configuration not found"
        assert hasattr(scenarios, 'generate_scenarios_from_config'), "generate_scenarios_from_config not found"
        assert hasattr(scenarios, 'render_pareto_frontiers'), "render_pareto_frontiers not found"
        assert hasattr(scenarios, 'render_scenario_comparison_table'), "render_scenario_comparison_table not found"
        assert hasattr(scenarios, 'render_sensitivity_analysis'), "render_sensitivity_analysis not found"
        
        print("✓ All required functions exist")
        return True
    except AssertionError as e:
        print(f"✗ Function check failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error checking functions: {e}")
        return False


def test_generate_scenarios():
    """Test scenario generation from config."""
    try:
        from pages import scenarios
        
        # Test gas price scenario generation
        config = {
            'type': 'gas_price',
            'variations': [0.5, 1.0, 1.5]
        }
        
        scenario_list = scenarios.generate_scenarios_from_config(config)
        
        assert len(scenario_list) == 3, f"Expected 3 scenarios, got {len(scenario_list)}"
        assert all('gas_price_multiplier' in s for s in scenario_list), "Missing gas_price_multiplier"
        
        print(f"✓ Generated {len(scenario_list)} gas price scenarios")
        
        # Test custom scenario generation
        config = {
            'type': 'custom',
            'gas_variations': [0.5, 1.0],
            'lmp_variations': [0.7, 1.0],
            'battery_variations': [350],
            'reliability_variations': [0.9999]
        }
        
        scenario_list = scenarios.generate_scenarios_from_config(config)
        
        assert len(scenario_list) == 4, f"Expected 4 scenarios, got {len(scenario_list)}"
        
        print(f"✓ Generated {len(scenario_list)} custom scenarios")
        
        return True
    except AssertionError as e:
        print(f"✗ Scenario generation test failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing scenario generation: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing scenarios page implementation...\n")
    
    tests = [
        ("Import Test", test_import),
        ("Function Structure Test", test_scenario_config_structure),
        ("Scenario Generation Test", test_generate_scenarios)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        results.append(test_func())
    
    print("\n" + "="*50)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
