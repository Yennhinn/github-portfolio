import pandas as pd
import numpy as np
import faiss
from time import time

t0 = time()

# 1. Load dữ liệu
scholarship = pd.read_csv('scholarship_sbert.csv')
user = pd.read_csv('user_final.csv')

# 2. Xử lý thời gian
user['created_at'] = pd.to_datetime(user['created_at'], errors='coerce')
scholarship['DateofPost'] = pd.to_datetime(scholarship['DateofPost'], errors='coerce')
scholarship['Deadline'] = pd.to_datetime(scholarship['Deadline'], errors='coerce')

# 3. Load embeddings
scholarship_matrix_all = np.load("sbert_matrix_scholarship.npy")
user_matrix = np.load("sbert_matrix_user.npy")

# 4. Normalize vectors
faiss.normalize_L2(scholarship_matrix_all)
faiss.normalize_L2(user_matrix)

# 5. Khởi tạo kết quả
results = []

# 6. Với từng user, lọc học bổng phù hợp theo ngày và tìm top K recommendation
K = 5
for i, row in user.iterrows():
    created_time = row['created_at']
    user_vec = user_matrix[i].reshape(1, -1)

    # Lọc học bổng có thời gian phù hợp
    mask = (scholarship['DateofPost'] <= created_time) & (scholarship['Deadline'] >= created_time)
    filtered_scholarship = scholarship[mask].reset_index()
    filtered_matrix = scholarship_matrix_all[mask.values]

    if len(filtered_scholarship) == 0:
        continue  # Không có học bổng phù hợp thời gian

    # FAISS index với subset
    faiss.normalize_L2(filtered_matrix)
    index = faiss.IndexFlatIP(filtered_matrix.shape[1])
    index.add(filtered_matrix)

    sim, idx = index.search(user_vec, min(K, len(filtered_scholarship)))

    for j in range(len(idx[0])):
        scholarship_id = filtered_scholarship.iloc[idx[0][j]]['ID']
        similarity = sim[0][j]
        results.append({
            'user_id': row['id'],
            'recommended_scholarship_id': scholarship_id,
            'similarity_score': round(similarity, 4)
        })

# 7. Kết quả ra DataFrame
print("----------sbert model----------")

recommendation_df = pd.DataFrame(results)
print(recommendation_df)


average_similarity = recommendation_df['similarity_score'].mean()
print("Average Similarity:", average_similarity)

from scipy.spatial.distance import cosine

user_distances = []

for user_id, group in recommendation_df.groupby('user_id'):
    scholarship_ids = group['recommended_scholarship_id'].tolist()

    if len(scholarship_ids) < 2:
        continue  # Cần ít nhất 2 học bổng để tính khoảng cách

    # Lấy index học bổng trong ma trận gốc
    indices = scholarship[scholarship['ID'].isin(scholarship_ids)].index.tolist()
    vectors = scholarship_matrix_all[indices]

    # Tính cosine distance giữa từng cặp học bổng
    pairwise_distances = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dist = cosine(vectors[i], vectors[j])
            pairwise_distances.append(dist)

    if pairwise_distances:
        avg_dist = np.mean(pairwise_distances)
        user_distances.append({
            'user_id': user_id,
            'avg_cosine_distance_topk': round(avg_dist, 4)
        })

# Tạo DataFrame
distance_df = pd.DataFrame(user_distances)
print("\n🔹 Top-K Cosine Distance per User:")
print(distance_df.head())

# Trung bình khoảng cách giữa học bổng của tất cả user
avg_cosine_distance_all = distance_df['avg_cosine_distance_topk'].mean()
print(f"\n🔹 Average Cosine Distance among Top-K Scholarships: {avg_cosine_distance_all:.4f}")





threshold = 0.6  # có thể điều chỉnh

# Đếm số user có ít nhất 1 học bổng với similarity >= threshold
good_recommendations = recommendation_df[recommendation_df['similarity_score'] >= threshold]
covered_user_ids = good_recommendations['user_id'].unique()

coverage = len(covered_user_ids) / user['id'].nunique()
print("Coverage (The proportion of users who receive relevant recommendations (>0.6):", coverage)




recommendation_df.to_csv('sim_sbert.csv', index=False)
distance_df.to_csv('distance_sbert.csv', index=False)

# 8. Tổng thời gian
print(f"\n✅ Recommendation done in {time() - t0:.3f} seconds")
