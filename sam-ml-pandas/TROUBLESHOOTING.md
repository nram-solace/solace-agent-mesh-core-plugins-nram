# Troubleshooting Guide for ML Pandas Agent

## Common Issues and Solutions

### 1. "Expecting value: line 1 column 1 (char 0)" Error

This error occurs when the `json_data` parameter is empty or not provided correctly.

**Possible Causes:**
- The `json_data` parameter is not being passed to the action
- The `json_data` parameter is an empty string
- The JSON data is malformed

**Solutions:**

1. **Check the action call:**
   ```python
   # Correct way to call the action
   load_data(
       load_type="json_data",
       json_data='[{"column1": "value1", "column2": "value2"}]',
       source_agent="sql_agent",
       description="Sales data from SQL query"
   )
   ```

2. **Test with sample JSON data:**
   ```python
   # Test with this sample data
   sample_json = '''
   [
       {"date": "2024-01-01", "region": "NY", "sales": 1000},
       {"date": "2024-01-02", "region": "NY", "sales": 1500},
       {"date": "2024-01-03", "region": "NY", "sales": 1200}
   ]
   '''
   ```

3. **Check JSON format:**
   - Ensure the JSON is valid (use a JSON validator)
   - Make sure it's a list of objects or has a `data`/`records`/`results` key
   - Avoid empty arrays

### 2. "Unclosed tags: invoke_action, parameter" Error

This error suggests there's an XML/JSON parsing issue in the response format.

**Possible Causes:**
- Malformed response from the action
- Issues with the ActionResponse format
- Problems with the data serialization

**Solutions:**

1. **Check the action response format:**
   ```python
   # Ensure proper ActionResponse format
   return ActionResponse(
       message="Success message",
       response_data=clean_result  # Must be JSON serializable
   )
   ```

2. **Validate data before returning:**
   ```python
   # Clean data for JSON serialization
   clean_result = agent.get_data_service().clean_data_for_json(result)
   ```

### 3. Testing the Data Loader Action

**Step 1: Test with file loading first:**
```python
# Test file loading
load_data(
    load_type="file",
    file_path="/path/to/test.csv",
    file_format="csv"
)
```

**Step 2: Test with simple JSON:**
```python
# Test with minimal JSON
load_data(
    load_type="json_data",
    json_data='[{"test": "value"}]',
    source_agent="test",
    description="Test data"
)
```

**Step 3: Test with realistic data:**
```python
# Test with realistic sales data
sales_json = '''
[
    {"date": "2024-01-01", "region": "NY", "product": "A", "sales": 1000},
    {"date": "2024-01-02", "region": "NY", "product": "B", "sales": 1500},
    {"date": "2024-01-03", "region": "NY", "product": "A", "sales": 1200}
]
'''

load_data(
    load_type="json_data",
    json_data=sales_json,
    source_agent="sql_agent",
    description="Sales data for NY region"
)
```

### 4. Debugging Steps

1. **Check the logs:**
   ```bash
   # Look for these log messages
   "Data loader action called with load_type: json_data"
   "JSON data length: X characters"
   "Source agent: sql_agent"
   ```

2. **Validate JSON manually:**
   ```python
   import json
   try:
       data = json.loads(your_json_string)
       print("JSON is valid")
   except json.JSONDecodeError as e:
       print(f"JSON error: {e}")
   ```

3. **Test DataFrame creation:**
   ```python
   import pandas as pd
   try:
       df = pd.DataFrame(your_data)
       print(f"DataFrame shape: {df.shape}")
   except Exception as e:
       print(f"DataFrame error: {e}")
   ```

### 5. Common JSON Formats Supported

The agent supports these JSON formats:

1. **List of records:**
   ```json
   [
       {"column1": "value1", "column2": "value2"},
       {"column1": "value3", "column2": "value4"}
   ]
   ```

2. **Data wrapper:**
   ```json
   {
       "data": [
           {"column1": "value1", "column2": "value2"},
           {"column1": "value3", "column2": "value4"}
       ]
   }
   ```

3. **Records wrapper:**
   ```json
   {
       "records": [
           {"column1": "value1", "column2": "value2"},
           {"column1": "value3", "column2": "value4"}
       ]
   }
   ```

4. **Results wrapper:**
   ```json
   {
       "results": [
           {"column1": "value1", "column2": "value2"},
           {"column1": "value3", "column2": "value4"}
       ]
   }
   ```

### 6. Getting Help

If you're still having issues:

1. **Check the agent logs** for detailed error messages
2. **Test with the sample JSON** provided above
3. **Verify your JSON format** matches one of the supported formats
4. **Ensure the json_data parameter** is not empty or null

The most common issue is that the `json_data` parameter is empty or not being passed correctly from the calling agent. 