SELECT name FROM sqlite_master WHERE type='table';

-- As a synthetic dataset, distributions are intentionally uniform
-- Analytical focus is on demonstrating SQL techniques (joins, window functions, CTEs, aggregations) 
  -- and risk segmentation logic rather than drawing real-world business conclusions.

SELECT 'customers' as table_name, COUNT(*) as row_count FROM customers
UNION ALL SELECT 'accounts', COUNT(*) FROM accounts
UNION ALL SELECT 'transactions', COUNT(*) FROM transactions
UNION ALL SELECT 'loans', COUNT(*) FROM loans
UNION ALL SELECT 'cards', COUNT(*) FROM cards
UNION ALL SELECT 'merchants', COUNT(*) FROM merchants
UNION ALL SELECT 'branches', COUNT(*) FROM branches;

-- Check null values
SELECT COUNT(*) - COUNT(customer_id) AS null_customer_id,
       COUNT(*) - COUNT(credit_score) AS null_credit_score
FROM customers;

-- Check duplicate transactions
SELECT transaction_id, COUNT(*)
FROM transactions
GROUP BY transaction_id
HAVING COUNT(*) > 1;

-- Check date range
SELECT MIN(transaction_date), MAX(transaction_date)
FROM transactions;


-- 1. Credit Risk Segmentation
-- 1.1. How are customers distributed across credit score segments, and what is the average balance per segment?
SELECT
    CASE
        WHEN credit_score >= 750 THEN 'Excellent (750+)'
        WHEN credit_score >= 700 THEN 'Good (700-749)'
        WHEN credit_score >= 650 THEN 'Fair (650-699)'
        WHEN credit_score >= 600 THEN 'Poor (600-649)'
        ELSE 'Very Poor (<600)'
    END AS segment,
    COUNT(*) AS customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS percentage,
    ROUND(AVG(credit_score), 0) AS avg_credit_score
FROM customers
GROUP BY segment
ORDER BY avg_credit_score DESC;

-- 1.2. Which customers have a low credit score (<600) but high loan amounts (top 5%)?
WITH percentile AS (
    SELECT
        loan_amount,
        NTILE(100) OVER (ORDER BY loan_amount) AS pct
    FROM loans),
threshold AS (
    SELECT MIN(loan_amount) AS top5_threshold
    FROM percentile
    WHERE pct >= 95)
SELECT DISTINCT loans.customer_id,
    customers.credit_score,
    loan_amount
FROM loans
JOIN customers ON customers.customer_id = loans.customer_id
WHERE credit_score < 600
AND loan_amount >= (SELECT top5_threshold FROM threshold)
ORDER BY loan_amount DESC;

-- 1.3. Among customers with multiple loans, how does total loan exposure compare to their credit score?
WITH loan_sumary AS (
    SELECT DISTINCT loans.customer_id,
        COUNT(loans.customer_id) AS number_of_loans,
        SUM(loan_amount) AS total_loan_amount,
        customers.credit_score
    FROM loans
    JOIN customers ON customers.customer_id = loans.customer_id
    GROUP BY loans.customer_id, customers.credit_score
    HAVING COUNT(loans.customer_id) > 1
    ORDER BY total_loan_amount DESC)
SELECT
    CASE
        WHEN credit_score >= 750 THEN 'Excellent (750+)'
        WHEN credit_score >= 700 THEN 'Good (700-749)'
        WHEN credit_score >= 650 THEN 'Fair (650-699)'
        WHEN credit_score >= 600 THEN 'Poor (600-649)'
        ELSE 'Very Poor (<600)'
    END AS credit_segment,
    AVG(credit_score) AS avg_credit_score,
    CAST(SUM(total_loan_amount) AS BIGINT) AS total_exposure
FROM loan_sumary
GROUP BY credit_segment
ORDER BY avg_credit_score DESC;


-- 2. Spending Pattern Analysis
-- 2.1. Which merchant categories drive the highest transaction volume and average spend?
SELECT
    m.merchant_name,
    COUNT(t.transaction_id) AS transaction_count,
    AVG(t.amount_usd) AS avg_spent
FROM merchants m
JOIN transactions t ON m.merchant_id = t.merchant_id
GROUP BY m.merchant_id
ORDER BY transaction_count DESC;

-- 2.2. Are there seasonal patterns in transaction activity — which months show the highest spikes?
SELECT
    strftime('%m', transaction_date) AS month,
    ROUND(SUM(CASE WHEN strftime('%Y', transaction_date) = '2019' THEN amount_usd ELSE 0 END), 2) AS y2019,
    ROUND(SUM(CASE WHEN strftime('%Y', transaction_date) = '2020' THEN amount_usd ELSE 0 END), 2) AS y2020,
    ROUND(SUM(CASE WHEN strftime('%Y', transaction_date) = '2021' THEN amount_usd ELSE 0 END), 2) AS y2022,
    ROUND(SUM(CASE WHEN strftime('%Y', transaction_date) = '2022' THEN amount_usd ELSE 0 END), 2) AS y2022,
    ROUND(SUM(CASE WHEN strftime('%Y', transaction_date) = '2023' THEN amount_usd ELSE 0 END), 2) AS y2023,
    ROUND(SUM(CASE WHEN strftime('%Y', transaction_date) = '2024' THEN amount_usd ELSE 0 END), 2) AS y2024,
    ROUND(SUM(CASE WHEN strftime('%Y', transaction_date) = '2025' THEN amount_usd ELSE 0 END), 2) AS y2025
FROM transactions
GROUP BY month
ORDER BY month;
SELECT
    strftime('%m', transaction_date) AS month,
    ROUND(SUM(amount_usd), 2) AS total_amount
FROM transactions
GROUP BY month
ORDER BY total_amount;

-- 2.3. Which customers show abnormal spending growth compared to their own historical average?

-- 3. Portfolio & Loan Exposure
-- 3.1. Which account types hold the highest average balance?
-- 3.2. Is there a correlation between interest rate and credit score?
-- 3.3. Which customers have outstanding loans but insufficient account balance to cover monthly payments — potential liquidity risk?

-- 4. Anomaly & Fraud Signals
-- 4.1. Which accounts show an unusually high number of transactions within a single day?
-- 4.2. Which transactions exceed 3x the customer's own average spending amount?
-- 4.3. Which cards are approaching expiration but still generating active transactions?

