# B2 — corpus hỗ trợ CNTT trong trường

Tệp `custom_campus_helpdesk.jsonl` chứa **250 ticket tiếng Việt**: 200 mẫu train và 50 mẫu validation. Năm nhãn intent (`cap_quyen`, `su_co`, `huong_dan`, `thay_doi`, `bao_mat`) có 50 mẫu mỗi nhãn. Schema giữ bốn trường của bài gốc (`intent`, `urgency`, `product`, `sentiment`); `product` ở miền này là tên dịch vụ hoặc thiết bị. Mỗi dòng có `instruction`, `input`, `output`, `label` và `split`.

## Nguồn và giới hạn chất lượng

Đây là **dữ liệu tổng hợp tự biên soạn**, không phải ticket thật và không chứa thông tin cá nhân thật. Tôi viết 50 tình huống nền riêng cho các dịch vụ như VLearn, email sinh viên, VPN, thư viện số, máy in và phòng lab; mỗi tình huống có năm cách diễn đạt. Nhãn intent lấy từ mục đích của tình huống; nhãn urgency và sentiment lấy từ ngôn ngữ thể hiện trong từng cách diễn đạt. Các tình huống bảo mật đều được gắn urgency `cao` vì cần phản ứng sớm. Mẫu được rà về tính nhất quán giữa câu chữ và nhãn; tuy nhiên dữ liệu tổng hợp có thể ít đa dạng hơn ticket thực tế và không thay thế kiểm định trên người dùng thật.

Tệp được tạo bằng `python scripts/make_custom_dataset.py`. Script kiểm tra đúng 250 câu khác nhau sau chuẩn hóa dấu câu/khoảng trắng và không trùng **nguyên văn sau chuẩn hóa** với ba tập đánh giá đóng băng của bài gốc. Cách kiểm tra này không chứng minh không có trùng ý; việc tách miền CNTT trường học khỏi ticket bán lẻ CSKH là hàng rào bổ sung. Tập validation gồm hai tình huống nền cuối của mỗi intent; toàn bộ năm biến thể của một tình huống nằm cùng một split, nên không có bản diễn đạt của cùng tình huống xuất hiện ở cả train và validation.

## Cách sử dụng và khử nhiễm

Corpus này là bài thử miền riêng tách biệt. Nó **không** thay `data/train_seed.jsonl`, `eval_target.jsonl`, `eval_regression.jsonl` hay checksum/phán quyết của NB1–NB5 đã nộp. Nếu huấn luyện thêm, chỉ dùng các dòng `split=train`; dùng `split=validation` để kiểm tra riêng và giữ nguyên holdout của core lab. Không được báo các điểm core hiện có như kết quả trên corpus này. Trước khi dùng dữ liệu thật, cần kiểm tra thêm trùng ý, sự cân bằng theo dịch vụ, câu hỏi mơ hồ và chất lượng nhãn bằng người chấm độc lập.
