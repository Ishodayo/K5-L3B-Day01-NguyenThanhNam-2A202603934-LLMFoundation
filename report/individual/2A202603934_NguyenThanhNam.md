# Báo cáo cá nhân --- Day 01

## Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603934
- **Họ và tên:** Nguyễn Thành Nam
- **SEED:** 8760

---

## 1. Bằng chứng

| Tệp | Trạng thái |
| :--- | :---: |
| `day01_lab.ipynb` --- Notebook đã chạy và còn output | ☑ |
| `lab1_report.json` --- Được sinh từ ô "Báo cáo cuối" | ☑ |
| `hand_calc.jpg` --- Ảnh tính tay có đầy đủ các bước | ☐ |

---

## 2. Kết quả chính

### 2.1. Chất lượng mô hình

| Mô hình | Precision | Recall | F1 (test) | F1 (SUB 40) |
| :--- | :---: | :---: | :---: | :---: |
| BoW + LogReg (tự code) | 0.986 | 0.737 | 0.844 | 0.857 |
| BoW + MLP (deep learning) | 0.956 | 0.869 | 0.910 | 0.947 |
| Encoder (MiniLM) + kNN | 0.898 | 0.980 | 0.937 | 1.000 |
| Encoder (MiniLM) + LogReg | 0.980 | 0.970 | 0.975 | 1.000 |
| Encoder (MiniLM) + MLP head | 0.960 | 0.960 | 0.960 | 1.000 |
| LLM nhỏ zero-shot | 0.952 | 1.000 | 0.976 | 0.976 |
| LLM lớn zero-shot | 1.000 | 1.000 | 1.000 | 1.000 |
| LLM nhỏ few-shot | 1.000 | 1.000 | 1.000 | 1.000 |

### 2.2. Tốc độ và chi phí

| Mô hình | Train (s) | Throughput (tin/s) | p50 (ms/tin) | p95 (ms/tin) | USD / 1.000 tin |
| :--- | :---: | :---: | :---: | :---: | :---: |
| BoW + LogReg (tự code) | 7.327 | 30,780.826 | 0.046 | 0.075 | 0.000 |
| BoW + MLP (deep learning) | 7.581 | 43,809.223 | 0.175 | 0.290 | 0.000 |
| Encoder (MiniLM) + kNN | 83.968 | 30.079 | 48.401 | 71.596 | 0.000 |
| Encoder (MiniLM) + LogReg | 84.440 | 46.804 | 39.488 | 62.616 | 0.000 |
| Encoder (MiniLM) + MLP head | 85.085 | 45.769 | 40.689 | 62.718 | 0.000 |
| LLM nhỏ zero-shot | 0.000 | 0.773 | 1,282.410 | 1,582.264 | 0.023 |
| LLM lớn zero-shot | 0.000 | 0.313 | 2,998.505 | 5,533.845 | 0.053 |
| LLM nhỏ few-shot | 0.000 | 0.898 | 1,133.227 | 1,391.339 | 0.063 |

**Lưu ý khi đọc kết quả:**
- Các hàng LLM chỉ được đo trên SUB (40 tin), vì vậy so sánh chất lượng của LLM bằng cột **F1 (SUB 40)**.
- SUB chỉ có 40 tin; một tin dự đoán sai có thể làm F1 thay đổi đáng kể. Chênh lệch dưới khoảng 0.05 F1 nên được xem là tương đối sát nhau, không nên xếp hạng chỉ dựa vào phần chênh lệch nhỏ.
- Inference được đo từ văn bản thô đến nhãn, bao gồm cả vectorize/encode. p50 và p95 là độ trễ khi xử lý từng tin như một request thực tế.
- Thời gian train của các mô hình MiniLM bao gồm thời gian encoder mã hóa tập train và thời gian train phần kNN/LogReg/MLP tương ứng. LLM không cần train trong thí nghiệm nên thời gian train được ghi là 0.
- Các số đo thời gian phụ thuộc vào phần cứng CPU/GPU của môi trường chạy notebook.

---

## 3. Dự đoán và thực tế

| Nội dung | Em dự đoán | Kết quả thực tế | Nhận xét |
| :--- | :--- | :--- | :--- |
| `emb_tricky` | Encoder + LogReg đoán đúng nhiều hơn | Encoder + kNN đạt F1 = 0.937, LogReg đạt 0.975 trên test thường | Dự đoán đúng hướng về khả năng của Embedding + LogReg trên tập test thường, nhưng chưa đủ dữ liệu để kết luận mô hình nào tốt hơn trên từng case khó. |
| `llm_fewshot` | Few-shot làm F1 tăng | Gemini nhỏ zero-shot đạt 0.976, few-shot đạt 1.000 | Đúng dự đoán trong lần thử nghiệm này, nhưng chưa thể khẳng định mức cải thiện luôn ổn định trên dữ liệu khác. |
| `hard_eval` | LLM khó nhất với Prompt Injection | LLM đạt 100% ở nhóm Prompt Injection; nhóm khó nhất là `ham_looks_spam` (80% zero-shot, 90% defensive) | Đây là dự đoán sai rõ nhất: trong bộ case khó này, LLM xử lý Prompt Injection tốt hơn dự kiến; nhóm tin ham có hình thức giống spam mới là nhóm khó hơn. |

---

## 4. Phân tích 50 case khó

### 4.1. Mức giảm hiệu suất
Mô hình giảm hiệu suất nhiều nhất là **BoW + LogReg**, với F1 giảm từ **0.857** trên tập test thông thường xuống **0.381** trên 50 case khó. BoW + MLP cũng giảm từ **0.947** xuống **0.435**.
Nguyên nhân là BoW chủ yếu dựa vào sự xuất hiện của từ khóa, nên khó nhận biết ngữ cảnh và ý nghĩa của câu khi người gửi thay đổi cách diễn đạt hoặc cố tình né từ khóa.

### 4.2. Những nhóm case khó nhất
| Nhóm mô hình | Nhóm case khó nhất | Kết quả |
| :--- | :--- | :---: |
| BoW + LogReg | Spam giọng tự nhiên (`natural_spam`) | 20% |
| BoW + MLP | Spam giọng tự nhiên (`natural_spam`) | 10% |
| Embedding + kNN | Nói về spam (`talks_about_spam`) | 25% |
| Embedding + LogReg | Thao túng / lừa đảo (`manipulation`) | 22% |
| Embedding + MLP head | Nói về spam (`talks_about_spam`) | 50% |
| LLM zero-shot | Tưởng spam nhưng không phải (`ham_looks_spam`) | 73% |
| LLM defensive | Tưởng spam nhưng không phải (`ham_looks_spam`) | 91% |

### 4.3. Hai case cụ thể minh họa
**Case #10 --- Tưởng spam nhưng không phải (`ham_looks_spam`)**
> Call 07700 900123 to reach me, my old phone died. New number!
Đây là tin nhắn hợp lệ thông báo đổi số điện thoại. Tuy nhiên, nội dung có số điện thoại, từ "Call" và dấu chấm than nên dễ bị nhầm thành spam.
**Case #12 --- Spam giọng tự nhiên (`natural_spam`)**
> hi there, sorry to bother you, my friend gave me your number, I'm new in town and looking for someone to chat with, check my profile at meetlocal.club
Đây là tin spam được viết bằng giọng lịch sự, thân thiện và không sử dụng những từ khóa quảng cáo phổ biến. Người đọc cần chú ý đến đường link và mục đích dụ người nhận truy cập.

## 5. Thử thách cá nhân

Với **SEED = 8760**, em được giao thử thách số 1: thay đổi ngưỡng quyết định (*threshold*) từ 0.5 thành 0.2 và 0.8 cho mô hình Logistic Regression ở Phần 1. Mục tiêu là tính Precision, Recall, F1 tại từng ngưỡng và lựa chọn ngưỡng phù hợp cho ngân hàng khi cần tránh chặn nhầm tin nhắn OTP hợp lệ.

### 5.1. Kết quả thực nghiệm

| Threshold | Precision | Recall | F1 |
| :---: | :---: | :---: | :---: |
| 0.2 | 0.864 | 0.899 | 0.881 |
| 0.5 | 0.986 | 0.737 | 0.844 |
| 0.8 | 1.000 | 0.485 | 0.653 |

### 5.2. Giải thích

- **Threshold = 0.2:** Mô hình dễ dự đoán tin nhắn là spam hơn, nên Recall cao (0.899), nhưng Precision thấp hơn (0.864), tức có nguy cơ chặn nhầm tin hợp lệ.
- **Threshold = 0.5:** Precision đạt 0.986 và Recall đạt 0.737, tạo sự cân bằng tương đối giữa việc phát hiện spam và hạn chế chặn nhầm.
- **Threshold = 0.8:** Mô hình chỉ phân loại spam khi xác suất dự đoán rất cao. Precision đạt 1.000, nhưng Recall giảm còn 0.485, nghĩa là bỏ sót nhiều tin spam hơn.

### 5.3. Kết luận

Em chọn **Threshold = 0.8** cho ngân hàng vì ưu tiên tránh chặn nhầm tin nhắn OTP hợp lệ. Trong tập test, Precision = 1.000 cho thấy không ghi nhận trường hợp dự đoán spam nhưng thực tế là ham. Tuy nhiên, ngưỡng này khiến mô hình bỏ sót nhiều spam, vì vậy cần cân nhắc thêm cơ chế kiểm tra đối với những tin nhắn chưa đủ độ tin cậy. Đây là kết quả trên tập test, không đảm bảo tuyệt đối khi triển khai thực tế.

---

## 6. Phương án cho sản phẩm thật

Với yêu cầu xử lý **10 triệu SMS/ngày** và độ trễ **p95 dưới 50 ms/tin**, em chọn **BoW + MLP** vì mô hình đáp ứng yêu cầu tốc độ với p95 = **0.290 ms**, throughput khoảng **43,809 tin/giây** và F1 = **0.910**. BoW + LogReg cũng đạt yêu cầu độ trễ với p95 = **0.075 ms**, nhưng chất lượng phân loại thấp hơn (F1 = **0.844**).

Em chấp nhận đánh đổi việc sử dụng BoW + MLP có chi phí và độ phức tạp cao hơn LogReg để đổi lấy chất lượng phân loại tốt hơn. LLM có F1 cao hơn nhưng p95 lớn hơn 50 ms và chi phí vận hành hàng tháng cao (Gemini nhỏ zero-shot khoảng **6,900 USD**, Gemini lớn khoảng **15,900 USD**, few-shot khoảng **18,900 USD**), nên chưa phù hợp để xử lý toàn bộ SMS.

---

## 7. AI log và điều đã học

### 7.1. AI log

### 7.2. Điều đã học

Qua bài thực hành, em hiểu rõ hơn sự khác nhau giữa BoW, Embedding và LLM trong bài toán phân loại spam. BoW có tốc độ xử lý nhanh nhưng dễ bị ảnh hưởng khi tin nhắn thay đổi cách diễn đạt. Embedding biểu diễn ngữ nghĩa tốt hơn nhưng có độ trễ cao hơn. LLM cho kết quả tốt trên bộ case khó trong lần thử nghiệm này, nhưng chi phí và độ trễ khiến việc áp dụng cho toàn bộ SMS chưa phù hợp.
Em cũng hiểu rằng lựa chọn mô hình cho sản phẩm thực tế không chỉ dựa vào F1 mà còn phải cân nhắc độ trễ, throughput, chi phí và hậu quả của việc phân loại nhầm.
