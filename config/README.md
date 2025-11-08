# Configuration Files

This directory contains configuration files for the datacenter energy optimization project.

## Gurobi WLS Configuration

### Setup Instructions

1. **Copy the example file:**
   ```bash
   cp config/gurobi_wls.json.example config/gurobi_wls.json
   ```

2. **Edit `config/gurobi_wls.json` with your credentials:**
   ```json
   {
     "WLSACCESSID": "your-wls-access-id",
     "WLSSECRET": "your-wls-secret",
     "LICENSEID": 12345
   }
   ```

3. **Get your WLS credentials:**
   - Go to: https://license.gurobi.com/manager/licenses
   - Log in with your Gurobi account
   - Find your WLS license
   - Copy the WLSACCESSID, WLSSECRET, and LICENSEID

### Testing Your Configuration

Run the configuration test script:

```bash
python src/optimization/gurobi_config.py
```

This will check:
- ✓ Config file exists
- ✓ Credentials are set
- ✓ Gurobi package is installed
- ✓ License is valid

### Security

The `config/gurobi_wls.json` file is automatically added to `.gitignore` and will **not** be committed to version control. This keeps your credentials secure.

### Automatic Loading

The optimization scripts automatically load your WLS credentials from this file. You don't need to set environment variables manually.

### Fallback Behavior

If Gurobi WLS is not configured or the license is invalid, the optimization scripts will automatically fall back to using the open-source GLPK solver. You'll see a message like:

```
Note: Using GLPK solver (open-source). For faster solving, install Gurobi.
```

### Troubleshooting

**Problem: "Config file not found"**
- Solution: Create `config/gurobi_wls.json` from the example file

**Problem: "Credentials not configured"**
- Solution: Fill in your actual WLS credentials in the JSON file

**Problem: "Gurobi license is invalid"**
- Solution: Verify your credentials at https://license.gurobi.com/manager/licenses
- Make sure your license is active and not expired

**Problem: "Gurobi Python package not installed"**
- Solution: Install with `pip install gurobipy`

### Alternative: Environment Variables

If you prefer to use environment variables instead of the config file, you can set:

```bash
export WLSACCESSID="your-wls-access-id"
export WLSSECRET="your-wls-secret"
export LICENSEID=12345
```

The config file takes precedence over environment variables if both are present.
