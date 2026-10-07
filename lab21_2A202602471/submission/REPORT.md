# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Hải Long  **MSSV**: 2A202602471  **Ngày**: 07/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `T4 16GB (Google Colab)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.
>
> **Mẫu này là gợi ý.** Bạn được tự chọn base model, dataset và tự viết report theo cấu
> trúc của mình — miễn là có đủ: lựa chọn + lý do, bằng chứng mask, mốc đóng băng, kết quả,
> phán quyết, điều học được (rubric 4.1).

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (mặc định) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 / 30 |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json: verdict = "reasoning preserved — safe to train on traces")*.
*Nếu không: bạn đã xử lý thế nào?* Template của dòng model `Qwen3.5` bảo tồn đầy đủ thẻ mở/đóng và phần thân của `<think>`, do đó chuỗi suy luận không bị nuốt mất trong quá trình tiền xử lý, không cần can thiệp regex hay sửa chat template thủ công. Về `max_length`, mặc dù p95 đo được là 98 token (gợi ý luỹ thừa 2 là 256), cấu hình Tier T4 giữ mức trần an toàn `1024` để đảm bảo không bao giờ bị cắt ngắn ngữ cảnh trong các tình huống câu hỏi thực tế dài hơn mà vẫn hoàn toàn vừa vặn trong ngưỡng VRAM.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 (41.49%) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```json
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3354.4 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1001.4 |
| (c) LoRA fine-tune | 0.970 | 0.544 | 1.000 | 1477.4 |

**(b) có thật sự mạnh hơn (a) không?** Có — Baseline (b) với prompt tối ưu đạt độ chính xác target 0.765 và format JSON hoàn hảo 1.000, trong khi baseline (a) ngây thơ đạt 0.000 ở cả hai tiêu chí do không ràng buộc được định dạng JSON 4 trường. Đồng thời độ trễ của (b) giảm đáng kể từ 3354.4 ms xuống 1001.4 ms nhờ greedy decoding sinh đúng cấu trúc ngắn gọn.
Bạn có sửa `OPTIMIZED_PROMPT` không? Nếu có: **làm mạnh lên hay yếu đi**, và vì sao? Không sửa, tôi giữ nguyên prompt tối ưu gốc được cung cấp (mã băm SHA: `719e74d3b6232053`) để đảm bảo tính liêm chính và công bằng tuyệt đối cho phép so sánh của toàn bộ pipeline.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6254 | 0.970 | 423.8 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 1e-4 | 0.5366 | 0.970 | 287.0 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | 0.000 | 431.6 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.940 | 508.6 | 3.86 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**
Trên tập target, `attn_only` đạt độ chính xác 0.970, hoàn toàn hoà với bản `correct` (0.970). Tuy nhiên, thứ tự này ngược với thứ tự theo train loss: `attn_only` có train loss thấp hơn rõ rệt (0.5366 so với 0.6254 của `correct`). Việc train loss thấp hơn nhưng không chuyển hóa thành độ chính xác kiểm tra cao hơn chứng minh rằng việc dồn toàn bộ ngân sách tham số vào ma trận attention với rank cực lớn ($r=283$) chỉ giúp mô hình ghi nhớ (overfit) dữ liệu huấn luyện nhanh hơn. Điều này khẳng định vị trí gắn adapter (`all-linear` trên toàn bộ decoder) ở rank vừa phải ($r=16$) mang lại biểu diễn tổng quát hóa tốt hơn và ổn định hơn so với việc cố tình đẩy rank lên mức cực đoan ở một vài module hẹp.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Đường loss của `wrong_lr` hầu như phẳng và dừng lại ở mức rất cao là 1.5702, hoàn toàn không giảm sâu như `correct` (0.6254), kéo theo điểm target và format trên tập đánh giá đều rơi về 0.000. Nếu chỉ nhìn vào đường loss phẳng lì này mà không biết bản chất learning rate đã bị hạ 10 lần (1e-5 so với 1e-4), người làm rất dễ kết luận sai lầm rằng cấu hình LoRA này không thể hội tụ hoặc phương pháp PEFT không có khả năng học tác vụ trích xuất thông tin tiếng Việt. Trên thực tế, đây chỉ là hậu quả kinh điển của Lỗi #2: áp dụng sai thang đo learning rate của full fine-tuning cho LoRA, bởi LoRA cần bước học lớn hơn (~10×) để cập nhật hiệu quả các ma trận xấp xỉ rank thấp.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
`qlora` 4-bit giúp tiết kiệm VRAM rất ấn tượng, cắt giảm dung lượng bộ nhớ từ 8.78 GB xuống chỉ còn 3.86 GB (tiết kiệm gần 56% VRAM đỉnh). Tuy nhiên, cái giá phải trả là thời gian huấn luyện kéo dài thêm (508.6s so với 423.8s của fp16 do overhead của kernel giải nén lượng tử 4-bit) và độ chính xác target bị tụt nhẹ từ 0.970 xuống 0.940. Số đo thực nghiệm này hoàn toàn ủng hộ khuyến nghị chính thức của nhà phát triển kiến trúc Qwen3.5: khi tài nguyên phần cứng (như GPU T4 16GB) vẫn đủ chứa mô hình ở dạng half-precision (fp16/bf16 LoRA chiếm ~8.78 GB), không nên đánh đổi chất lượng mô hình lấy dung lượng bằng QLoRA 4-bit, bởi sai số lượng tử hóa làm suy hao hiệu năng trích xuất trường văn bản.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.247` · `valid_trace_rate = 0.0`

Diễn giải (≥100 từ). Nếu FAILED: **vì sao**, và điều đó nói gì về bài toán của bạn?
Bản fine-tune LoRA đã tạo ra một bước nhảy vọt ấn tượng trên tác vụ mục tiêu với target tăng thêm +0.205 (từ 0.765 của baseline b lên 0.970) và format JSON đạt chuẩn tuyệt đối 1.000. Tuy nhiên, cổng hồi quy đánh giá chung cuộc là FAILED do năng lực tổng quát (regression) bị sụt giảm nặng nề tới -0.247 (từ 0.791 xuống 0.544, vượt rất xa ngưỡng dung sai tối đa 0.020). 

Nguyên nhân trực tiếp dẫn tới hiện tượng này là "quên thảm họa" (catastrophic forgetting): khi huấn luyện mô hình 4B tham số trên một tập dữ liệu nhỏ (225 mẫu) chỉ toàn định dạng JSON của bài toán CSKH mà không có bất kỳ dữ liệu đối chứng tổng quát nào đi kèm, các trọng số LoRA đã bị thiên lệch hoàn toàn về phân phối chuyên biệt này. Kết quả này phản ánh bản chất thực tế trong công nghiệp: việc fine-tune một tác vụ hẹp nếu không có cơ chế bảo toàn tri thức (chẳng hạn như trộn 1–5% replay data kiến thức chung theo deck §6.3) sẽ làm hỏng năng lực đàm thoại và suy luận phổ quát của nền tảng LLM gốc.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt. | intent: doi_tra, urgency: cao, product: chuột không dây, sentiment: tich_cuc | Đúng format, nhưng phân loại sai trường urgency | Đúng hoàn toàn 4 trường (score 1.0) | ✅ **FT thắng**: Nhận diện chuẩn xác tính cấp thiết "Gấp" và cảm xúc tích cực dù khách yêu cầu đổi trả. |
| 2 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé. Bực mình. | intent: hoan_tien, urgency: trung_binh, product: ốp lưng điện thoại, sentiment: tieu_cuc | Sai trường sentiment (nhầm sang trung_tinh) | Đúng hoàn toàn 4 trường (score 1.0) | ✅ **FT thắng**: Bắt đúng cảm xúc "tieu_cuc" từ từ khóa "Bực mình" và ý định "hoan_tien". |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều. | intent: hoan_tien, **urgency: thap**, product: bình giữ nhiệt, sentiment: tich_cuc | Nhận diện đúng urgency: thap | Đoán sai **urgency: trung_binh** (score 0.75) | ❌ **FT thua**: Khách hỏi lịch sự "Khi nào tiện", nhưng model FT bị over-estimate mức độ cấp bách thành trung bình. |
| 4 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi. | intent: san_pham_loi, **urgency: thap**, product: nồi chiên không dầu, sentiment: trung_tinh | Nhận diện đúng urgency: thap | Đoán sai **urgency: trung_binh** (score 0.75) | ❌ **FT thua**: Tương tự ca 3, model fine-tune không nhạy với cụm từ giảm nhẹ mức độ khẩn cấp "Khi nào tiện". |
| 5 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop nhiều. | intent: san_pham_loi, **urgency: thap**, product: áo khoác gió, sentiment: tich_cuc | Đúng ngữ cảnh lịch sự | Đoán sai **urgency: trung_binh** (score 0.75) | ❌ **FT thua**: Nhầm lẫn giữa lỗi sản phẩm thông thường và khiếu nại gấp, nâng urgency lên trung_binh. |

**Có mẫu chung nào ở các ca FT thua không?**
Có một mẫu chung rất rõ rệt ở tất cả các ca fine-tune bị thua (score 0.75): Mô hình fine-tune có xu hướng thiên lệch (bias) dự đoán nhầm độ cấp bách (`urgency`) từ `thap` lên `trung_binh`. Khi khách hàng sử dụng các cụm từ thể hiện sự thong thả như "Khi nào tiện", prompt tối ưu (b) của base model hiểu được ngữ nghĩa sắc thái này và gán đúng nhãn `thap`, trong khi bản fine-tune dường như đã bị "quá nhạy cảm" với các từ khóa khiếu nại (bị lỗi, chưa thấy tiền, thiếu phụ kiện) và luôn gán mức khẩn cấp tối thiểu là `trung_binh`.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** 
Nếu xét dưới góc độ một hệ thống triển khai toàn diện phục vụ đa mục đích, ta **chưa nên deploy** bản fine-tune này ngay lập tức vào production, bởi vì mô hình đã không vượt qua cổng kiểm soát hồi quy (năng lực kiến thức tổng quát sụt giảm mạnh -0.247). Tuy nhiên, nếu hệ thống được thiết kế theo kiến trúc microservices chuyên biệt — nơi model chỉ đóng vai trò một module backend biệt lập nhận văn bản ticket và trả về JSON phân loại để định tuyến yêu cầu cho nhân viên hỗ trợ — thì bản fine-tune này lại là một ứng viên xuất sắc: nó đạt độ chính xác tác vụ mục tiêu lên tới 97.0% (vượt trội hoàn toàn so với baseline prompt tối ưu 76.5%), định dạng JSON chuẩn xác 100%, và tốc độ xử lý nhanh chóng (1.4s). 

Đòn bẩy kỹ thuật thực sự quyết định thành công trong lab này không phải là việc cố gắng tăng rank lên mức cực đại (thể hiện qua việc `attn_only` rank 283 chỉ hòa điểm target với `correct` rank 16), mà nằm ở: (1) che loss mask chính xác (`assistant-only`) để không dạy mô hình học vẹt prompt, (2) chọn đúng thang learning rate cho LoRA (1e-4 thay vì thang 1e-5 của full-FT), và (3) phân bổ adapter toàn diện trên mọi lớp tuyến tính của text decoder (`all-linear`). Để sẵn sàng cho việc đưa vào vận hành thực tế không có rủi ro, bước tiếp theo là tiến hành huấn luyện lại với việc trộn bổ sung 1–5% dữ liệu tổng quát (replay data) nhằm khắc phục triệt để hiện tượng quên thảm họa.

**Ba điều tôi học được** (cụ thể, không generic):
1. **Rank không phải là đòn bẩy vạn năng**: Thí nghiệm đối chứng giữa `attn_only` ($r=283$) và `correct` ($r=16$) với cùng 32.46M tham số đã chứng minh rõ ràng: rank cao chỉ làm giảm train loss giả tạo (0.5366 vs 0.6254) do overfit, chứ không giúp ích cho bài toán đánh giá thực tế. Vị trí gắn adapter bao phủ toàn bộ text-linear mới là yếu tố quyết định chất lượng biểu diễn.
2. **LoRA cần thang Learning Rate riêng biệt**: Sai lầm hạ LR xuống mức 1e-5 (thang đo full-FT) trong run `wrong_lr` đã làm đóng băng quá trình học (loss kẹt ở 1.57, accuracy 0%), cho thấy LoRA bắt buộc phải dùng LR cao hơn khoảng 10 lần so với full fine-tuning để bù đắp cho việc cập nhật ma trận rank thấp.
3. **Cổng hồi quy 4 nhóm là công cụ bảo hiểm bắt buộc**: Nếu chỉ đo đạc dựa trên accuracy của tác vụ CSKH (97.0%) hoặc train loss, ta sẽ ngộ nhận rằng việc huấn luyện thành công rực rỡ. Việc đo thêm nhóm câu hỏi tổng quát (regression) đã bóc trần hiện tượng quên thảm họa (-0.247), điều mà perplexity hay accuracy cục bộ không bao giờ chỉ ra được.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Tôi sẽ tạo một bộ dữ liệu trộn gồm 250 mẫu CSKH kết hợp với 15–20 mẫu chỉ dẫn đàm thoại tổng quát (tương đương ~5% replay data) và huấn luyện lại cấu hình `correct`, sau đó chạy lại cổng hồi quy để chứng minh rằng mô hình vừa giữ vững target ≥ 0.95 vừa đưa `regression Δ` về mức an toàn (≥ -0.02) để chuyển phán quyết sang `PASSED`.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [x] B5 HuggingFace Hub — link: https://huggingface.co/long2711/qwen3.5-cs-triage-lora
