## 🚀 Cài đặt & Chạy nhanh

### 1. Clone repository
```bash
git clone https://github.com/NguyenQuocKhanh0/VoxCPM_infer.git
cd VoxCPM_infer
```
### 2. Cài đặt (recommend python 3.11)
```bash
python setup.py install
```
### 3. Chạy chương trình
```bash
python main.py
```
### 4. Những hạn chế đã biết + Lưu ý:
- Chất lượng âm thanh chưa ổn định, đôi khi có các hiện tượng mất chữ phía cuối, nhịp điệu không đều
- Không hỗ trợ điều chỉnh thời lượng âm thanh
- Đối với các đoạn dài, code tự động chia nhỏ thành các đoạn ngắn, có thể có hiện tượng giọng, nhịp điệu không đồng nhất giữa các đoạn
- Chất lượng âm thanh có thể phụ thuộc vào audio đầu vào
- Nếu không có Audio đầu vào, giọng sẽ random
- Model sẽ tự động phát hiện tiếng Anh và tiếng Việt, nếu phát hiện sai sót, có thể đặt text dạng "[vi]xin chào bạn, hôm nay là [en]tuesday [vi], ngày mai là thứ ba. Nếu cần fix cứng tiếng anh có thể đặt [en]before the English section[vi] nhé"
- Có thể giảm Steps xuống 10 để nhanh hơn
