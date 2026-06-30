# Model Comparison Report

Comparison of evaluation results for: **phi3, gemma3**.

## Overall Metrics

### Latency (seconds)

| Metric | phi3 | gemma3 |
| --- | --- | --- |
| avg latency | 11.533 | 12.099 |
| median latency | 8.494 | 8.914 |
| min latency | 5.369 | 4.994 |
| max latency | 57.861 | 105.177 |

### Answer Quality

| Metric | phi3 | gemma3 |
| --- | --- | --- |
| avg answer length words | 130.500 | 119.533 |
| avg sources | 1.733 | 1.733 |

## Category Metrics

### Average Latency by Category

| Category | phi3 | gemma3 |
| --- | --- | --- |
| budget | 9.297 | 7.444 |
| disease_burden | 8.681 | 11.088 |
| infrastructure | 8.010 | 8.672 |
| maternal_health | 8.910 | 8.624 |
| policy | 14.942 | 8.598 |
| workforce | 19.360 | 28.165 |

### Average Answer Length by Category (words)

| Category | phi3 | gemma3 |
| --- | --- | --- |
| budget | 175.400 | 65.200 |
| disease_burden | 80.200 | 165.400 |
| infrastructure | 119.600 | 118.200 |
| maternal_health | 92.000 | 133.600 |
| policy | 123.800 | 143.200 |
| workforce | 192.000 | 91.600 |

## Generated Artifacts

- Metrics CSV: `C:/Users/rohit/healthcare-ai/evaluation/reports/comparison_metrics.csv`
- Latency plot: `C:/Users/rohit/healthcare-ai/evaluation/reports/latency.png`
- Answer length plot: `C:/Users/rohit/healthcare-ai/evaluation/reports/answer_length.png`
- Category latency plot: `C:/Users/rohit/healthcare-ai/evaluation/reports/category_latency.png`
- Category answer length plot: `C:/Users/rohit/healthcare-ai/evaluation/reports/category_answer_length.png`
