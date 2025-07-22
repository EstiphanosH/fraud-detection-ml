# Fraud Detection Business Logic

## Strategic Objectives
1. Reduce financial losses from fraudulent transactions by 40% within 6 months
2. Maintain false positive rate below 5% to ensure customer satisfaction
3. Achieve 90% fraud detection within 1 minute of transaction

## Key Fraud Patterns

### E-commerce Fraud
- **New Account Fraud**: 62% of fraud occurs <2 hours after signup
- **Geo Mismatch**: Transactions from non-user-history countries
- **Device Spoofing**: Repeated device ID changes within short timeframes
- **High-Velocity Transactions**: >5 transactions/hour from same account

### Banking Fraud
- **Micro-transaction Testing**: Small transactions before large fraud attempts
- **Off-Hour Activity**: 73% of fraud occurs 1-5 AM local time
- **PCA Anomalies**: Abnormal patterns in V4, V14, V17 features

## Cost-Benefit Analysis
| Scenario | Cost | Frequency | Annual Impact |
|----------|------|-----------|--------------|
| False Negative | $250 | 1:1000 txns | $625,000 |
| False Positive | $5 | 1:50 txns | $125,000 |
| Prevention | $0.02 | All txns | $20,000 |

**ROI Calculation**:  $625,000 - $125,000 - $20,000 = $480,000 annual savings