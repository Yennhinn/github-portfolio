-- ===================================================================================================================================
--  1. Overview
---- Loan overview
SELECT
    COUNT(DISTINCT id) AS number_of_loans,
    SUM(loan_amount) AS total_loan_amount,
    ROUND(AVG(loan_amount), 1) AS avg_loan_amount
FROM loans;
---- 38,576 loans | $435M total | $11,296 avg loan size


---- How are loans distributed by status?
SELECT
    loan_status,
    SUM(loan_amount) AS total_loan_amount,
    ROUND(SUM(loan_amount) * 100.0 / SUM(SUM(loan_amount)) OVER (), 2) AS pct_of_amount,
    COUNT(*) AS number_of_loans,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_loans
FROM loans
GROUP BY loan_status;
---- 83% Fully Paid → portfolio is healthy overall
---- Charged Off rate 13.8% → above industry benchmark (3–5%), signals elevated credit risk


---- Which loan purpose has the highest volume, and how does it break down by status?
SELECT
    purpose,
    SUM(loan_amount) AS total_loan_amount,
    ROUND(SUM(loan_amount) * 100.0 / SUM(SUM(loan_amount)) OVER (), 2) AS pct_of_amount,
    SUM(CASE WHEN loan_status = 'Fully Paid'  THEN loan_amount ELSE 0 END) AS fully_paid,
    SUM(CASE WHEN loan_status = 'Charged Off' THEN loan_amount ELSE 0 END) AS charged_off,
    SUM(CASE WHEN loan_status = 'Current'     THEN loan_amount ELSE 0 END) AS current
FROM loans
GROUP BY purpose
ORDER BY total_loan_amount DESC;
-- Debt consolidation dominates at 53% → borrowers primarily refinancing existing debt
-- Small business: low volume but disproportionately high Charged Off amount → highest risk segment



-- ===================================================================================================================================
-- 2. Risk Segmentation  
---- Which grade and sub-grade has the highest Charged Off rate?
SELECT
    grade,
    sub_grade,
    SUM(loan_amount) AS total_loan_amount,
    SUM(CASE WHEN loan_status = 'Charged Off' THEN loan_amount ELSE 0 END) AS charged_off,
    ROUND(SUM(CASE WHEN loan_status = 'Charged Off' THEN loan_amount ELSE 0 END) * 100.0 / SUM(loan_amount), 2) AS pct_charged_off,
    COUNT(*) AS number_of_loans,
    SUM(CASE WHEN loan_status = 'Charged Off' THEN 1 ELSE 0 END) AS number_of_charged_off,
    ROUND(SUM(CASE WHEN loan_status = 'Charged Off' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS pct_of_loans
FROM loans
WHERE loan_status IN ('Fully Paid', 'Charged Off')
GROUP BY grade, sub_grade
ORDER BY grade, sub_grade
---- Grade A: 2–8%. Grade F–G: 30–48% → strong risk gradient


---- Does DTI differ significantly between Charged Off vs Fully Paid borrowers?
SELECT
    loan_status,
    ROUND(AVG(dti), 2) AS avg_dti,
    MIN(dti) AS min_dti,
    MAX(dti) AS max_dti,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY dti), 2) AS median_dti
FROM loans
WHERE loan_status IN ('Fully Paid', 'Charged Off')
GROUP BY loan_status
---- DTI: Charged Off (0.14) vs Fully Paid (0.13) → minimal difference, weak predictor

  
---- Does verification status have any impact on Charged Off rate?
SELECT
    verification_status,
    COUNT(*) AS number_of_loans,
    SUM(CASE WHEN loan_status = 'Charged Off' THEN 1 ELSE 0 END) AS number_of_charged_off,
    SUM(CASE WHEN loan_status = 'Charged Off' THEN loan_amount ELSE 0 END) AS total_charged_off,
    SUM(CASE WHEN loan_status = 'Fully Paid'  THEN loan_amount ELSE 0 END) AS total_fully_paid,
    ROUND(SUM(CASE WHEN loan_status = 'Charged Off' THEN loan_amount ELSE 0 END) * 100.0 / SUM(loan_amount), 2) AS pct_charged_off
FROM loans
WHERE loan_status IN ('Fully Paid', 'Charged Off')
GROUP BY verification_status
---- Verified loans have higher Charged Off rate (17.8%) than Not Verified (13%) → verification ≠ lower risk


  
-- ===================================================================================================================================
Phase 3 — Borrower Profile
annual_income và emp_length của nhóm default vs fully paid khác nhau thế nào?
home_ownership có tương quan với khả năng trả nợ không?
State nào có default rate cao nhất?


Phase 4 — Trend Analysis
Theo thời gian thì sao?

Loan volume theo tháng (issue_date) — MTD, MoM growth
Default rate có tăng theo thời gian không?
int_rate trung bình thay đổi theo tháng như thế nào?


Phase 5 — Advanced
Đào sâu hơn

Ranking: top 10 states by total loan amount dùng RANK()
Running total loan amount theo tháng dùng SUM() OVER
So sánh installment vs total_payment — ai đang trả ít hơn expected? (LAG())
