# Collaborative Workflow Example: SQL Agent + ML Pandas Agent

This example demonstrates how the ML Pandas agent works with other agents to get real data and provide analysis.

## Scenario: "Summarize sales in last 3 months from NY"

### Step 1: SQL Agent Gets the Data

**User Request:** "Get sales data for last 3 months from NY region"

**SQL Agent Response:**
```json
{
  "data": [
    {
      "date": "2024-01-15",
      "region": "NY",
      "product": "Product A",
      "quantity": 50,
      "unit_price": 25.99,
      "total_sales": 1299.50,
      "customer_id": "C001"
    },
    {
      "date": "2024-02-20",
      "region": "NY", 
      "product": "Product B",
      "quantity": 30,
      "unit_price": 45.00,
      "total_sales": 1350.00,
      "customer_id": "C002"
    }
    // ... more records
  ]
}
```

### Step 2: ML Pandas Agent Receives and Processes

**User Request:** "Load this data and summarize sales"

**ML Pandas Agent Actions:**

1. **Load Data:**
   ```python
   # Uses load_data action with json_data parameter
   load_data(
       load_type="json_data",
       json_data="[SQL_AGENT_RESULT_JSON]",
       source_agent="sql_agent",
       description="Sales data for NY region, last 3 months"
   )
   ```

2. **Summarize Data:**
   ```python
   # Uses summarize_data action for quick analysis
   summarize_data(
       summary_type="key_metrics",
       focus_columns="total_sales,quantity,unit_price",
       include_visualization=True
   )
   ```

### Step 3: ML Pandas Agent Response

**Summary Output:**
```
# Data Summary (Key Metrics)

**Source Agent:** sql_agent
**Description:** Sales data for NY region, last 3 months
**Data Shape:** 1,247 rows × 7 columns

### Numerical Metrics:
**total_sales:**
  - Mean: $1,234.56
  - Median: $1,100.00
  - Std: $567.89
  - Range: $50.00 to $5,000.00
  - Sum: $1,539,456.78

**quantity:**
  - Mean: 45.2
  - Median: 40.0
  - Std: 25.1
  - Range: 1 to 200
  - Sum: 56,324

### Key Insights:
- Total sales in NY region: $1,539,456.78
- Average order value: $1,234.56
- Total orders: 1,247
- Average quantity per order: 45.2 units
```

## Benefits of This Approach

1. **Real Data:** No canned or generated data - works with actual data from your systems
2. **Specialized Agents:** Each agent does what it does best
   - SQL Agent: Database queries and data retrieval
   - ML Pandas Agent: Data analysis and summarization
3. **Flexible Workflows:** Can work with any agent that returns JSON data
4. **Scalable:** Easy to add more agents to the workflow

## Supported Agent Integrations

- **SQL Database Agent:** For structured data queries
- **MongoDB Agent:** For document database queries
- **RAG Agent:** For document analysis and retrieval
- **Any Custom Agent:** That can return JSON data

## Example Workflow Commands

```bash
# 1. Get data from SQL agent
"Get sales data for last 3 months from NY region"

# 2. Load and summarize with ML pandas agent
"Load the SQL agent results and summarize sales metrics"

# 3. Perform detailed analysis
"Run a complete data analysis on the sales data"

# 4. Build ML model
"Train a regression model to predict total_sales"
```

This collaborative approach ensures you get real insights from your actual data, not synthetic examples! 