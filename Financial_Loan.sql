Phase 1 — Portfolio Overview
Hiểu tổng quan trước khi đi sâu

Tổng số loan, tổng loan amount, average loan amount là bao nhiêu?
Phân bổ theo loan_status (Fully Paid / Charged Off / Current) như thế nào?
Loan tập trung ở purpose nào nhiều nhất?


Phase 2 — Risk Segmentation
Ai đang có risk cao?

Default rate theo grade và sub_grade — grade nào nguy hiểm nhất?
dti của nhóm default có khác biệt rõ so với nhóm fully paid không?
verification_status có ảnh hưởng đến default rate không?


Phase 3 — Borrower Profile
Người vay trông như thế nào?

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
