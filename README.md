# CS313 Data Mining — TCGA-BRCA

Kho lưu trữ này tập trung vào phân tích dữ liệu TCGA-BRCA, bao gồm:

- **Streamlit clinical dashboard** để khám phá dữ liệu clinical JSON từ GDC.
- **Streamlit survival demo** để chấm điểm nguy cơ sống còn trên một mẫu bằng mô hình Cox.
- **Notebook/pipeline** cho các bước chuẩn hoá dữ liệu, chọn đặc trưng và huấn luyện mô hình.
- **Artifact và bảng đặc trưng** phục vụ inference/đánh giá.

## Cài đặt nhanh

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Lưu ý: nếu chạy `task3_demo.py`, cần thêm `scikit-survival` (chưa nằm trong `requirements.txt`).

## Chạy các ứng dụng Streamlit

### 1) Clinical dashboard

```bash
streamlit run app.py
```

- Input: JSON từ GDC (*Repository → Clinical*), ví dụ: `sample/clinical.cart.2025-10-21.json`.
- Output: thống kê cohort, biểu đồ nhân khẩu học, phân bố stage, receptor status và Kaplan–Meier OS.

### 2) Survival demo (Task 3)

```bash
streamlit run app_task3_demo.py
```

- Upload gene expression file (TSV/CSV) và điền các đặc trưng lâm sàng theo form.
- Dùng các artifact mặc định: `coxph_streamlit_artifact.joblib`, `coxnet_streamlit_artifact.joblib`.

## Cấu trúc thư mục chính

- `app.py`: Clinical dashboard Streamlit.
- `app_task3_demo.py`: Demo chấm điểm sống còn trên một mẫu.
- `task3_demo.py`: Script thử nghiệm/so sánh pipeline cho Task 3.
- `utils_clinical.py`: Hàm parse clinical JSON + feature engineering.
- `prepare_data/`: Notebook tiền xử lý và gộp dữ liệu.
- `train_and_preprocess/`: Pipeline/Notebook huấn luyện mô hình.
- `model_export/`: Mô hình phân loại đã xuất (normal/tumor, subtyping).
- `*.csv`: Bảng đặc trưng đã trích xuất (XGB/RF/SVM).
- `sample/`: Dữ liệu mẫu để chạy nhanh.

## Gợi ý dữ liệu đầu vào

- **Clinical JSON**: xuất từ GDC, dùng cho `app.py`.
- **Expression TSV/CSV**: xuất từ STAR/HTSeq hoặc bảng gene expression tương tự, dùng cho demo Task 3.

## Ghi chú

- Repo này chứa nhiều notebook và artifact để tham khảo; bạn có thể chạy lại pipeline trong `prepare_data/` và `train_and_preprocess/` nếu cần tái huấn luyện.
- Nếu thiếu dependency khi chạy notebook/streamlit, hãy cài bổ sung theo lỗi hiển thị.
