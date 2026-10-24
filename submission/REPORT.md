# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Hải Long  **MSSV**: 2A202602471
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `T4 16GB (Google Colab)`

Các số liệu trong báo cáo được đối chiếu với artifact trong `results/`.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (mặc định) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 / 30 |

Template của `Qwen3.5` giữ đầy đủ thẻ mở/đóng và phần thân `<think>` (`results/template_check.json`: `reasoning preserved — safe to train on traces`). Vì vậy tôi không sửa chat template hoặc xóa trace bằng regex. p95 đo được là 98 token; tier T4 vẫn dùng `max_length=1024` để có khoảng đệm cho ticket dài hơn và nằm trong ngân sách VRAM.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 (41.49%) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đầu đoạn token được tính loss:

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

Baseline (b) với prompt tối ưu đạt target 0.765 và format JSON 1.000, trong khi baseline (a) đạt 0.000 ở cả hai tiêu chí vì prompt không ràng buộc JSON bốn trường. Độ trễ của (b) giảm từ 3354.4 ms xuống 1001.4 ms do greedy decoding sinh câu trả lời ngắn gọn. Tôi giữ nguyên `OPTIMIZED_PROMPT` được cung cấp (mã băm SHA: `719e74d3b6232053`) để phép so sánh không bị thay đổi giữa các run.

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
Trên tập target, `attn_only` và `correct` đều đạt 0.970 dù `attn_only` có train loss thấp hơn (0.5366 so với 0.6254). Vì hai bản được khớp ngân sách tham số, số liệu này cho thấy giảm loss bằng cách tăng rank chỉ ở attention không đem lại lợi ích target trong phép so sánh này. Nó gợi ý vị trí gắn `text-linear` quan trọng hơn việc dồn ngân sách vào attention; riêng các số đo này chưa đủ để khẳng định mô hình đã overfit hoặc tổng quát hóa tốt hơn trên mọi dữ liệu. B4 kiểm tra vai trò của rank một cách riêng biệt bằng cách giữ cố định vị trí `text-linear`.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Đường loss của `wrong_lr` hầu như phẳng và dừng lại ở mức rất cao là 1.5702, hoàn toàn không giảm sâu như `correct` (0.6254), kéo theo điểm target và format trên tập đánh giá đều rơi về 0.000. Nếu chỉ nhìn vào đường loss phẳng lì này mà không biết bản chất learning rate đã bị hạ 10 lần (1e-5 so với 1e-4), người làm rất dễ kết luận sai lầm rằng cấu hình LoRA này không thể hội tụ hoặc phương pháp PEFT không có khả năng học tác vụ trích xuất thông tin tiếng Việt. Trên thực tế, đây chỉ là hậu quả kinh điển của Lỗi #2: áp dụng sai thang đo learning rate của full fine-tuning cho LoRA, bởi LoRA cần bước học lớn hơn (~10×) để cập nhật hiệu quả các ma trận xấp xỉ rank thấp.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
`qlora` 4-bit giúp tiết kiệm VRAM rất ấn tượng, cắt giảm dung lượng bộ nhớ từ 8.78 GB xuống chỉ còn 3.86 GB (tiết kiệm gần 56% VRAM đỉnh). Tuy nhiên, cái giá phải trả là thời gian huấn luyện kéo dài thêm (508.6s so với 423.8s của fp16 do overhead của kernel giải nén lượng tử 4-bit) và độ chính xác target bị tụt nhẹ từ 0.970 xuống 0.940. Số đo thực nghiệm này hoàn toàn ủng hộ khuyến nghị chính thức của nhà phát triển kiến trúc Qwen3.5: khi tài nguyên phần cứng (như GPU T4 16GB) vẫn đủ chứa mô hình ở dạng half-precision (fp16/bf16 LoRA chiếm ~8.78 GB), không nên đánh đổi chất lượng mô hình lấy dung lượng bằng QLoRA 4-bit, bởi sai số lượng tử hóa làm suy hao hiệu năng trích xuất trường văn bản.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.247` · `valid_trace_rate = 0.0`

Bản fine-tune được đánh giá theo cổng hồi quy đã đóng băng trước khi train:
Bản fine-tune LoRA đã tạo ra một bước nhảy vọt ấn tượng trên tác vụ mục tiêu với target tăng thêm +0.205 (từ 0.765 của baseline b lên 0.970) và format JSON đạt chuẩn tuyệt đối 1.000. Tuy nhiên, cổng hồi quy đánh giá chung cuộc là FAILED do năng lực tổng quát (regression) bị sụt giảm nặng nề tới -0.247 (từ 0.791 xuống 0.544, vượt rất xa ngưỡng dung sai tối đa 0.020). 

Nguyên nhân trực tiếp dẫn tới hiện tượng này là "quên thảm họa" (catastrophic forgetting): khi huấn luyện mô hình 4B tham số trên một tập dữ liệu nhỏ (225 mẫu) chỉ toàn định dạng JSON của bài toán CSKH mà không có bất kỳ dữ liệu đối chứng tổng quát nào đi kèm, các trọng số LoRA đã bị thiên lệch hoàn toàn về phân phối chuyên biệt này. Kết quả này phản ánh bản chất thực tế trong công nghiệp: việc fine-tune một tác vụ hẹp nếu không có cơ chế bảo toàn tri thức (chẳng hạn như trộn 1–5% replay data kiến thức chung theo deck §6.3) sẽ làm hỏng năng lực đàm thoại và suy luận phổ quát của nền tảng LLM gốc.

### 5.1 Thí nghiệm bổ sung sau NB5: replay tổng quát

Tôi giữ nguyên baseline (b), 50 câu target, 15 câu regression và 30 optimizer steps của NB3. Tám rồi 11 mẫu chỉ dẫn tổng quát được trộn vào 225 mẫu train theo seed 42; script kiểm tra tỷ lệ 1–5% và cấm trùng nguyên văn với holdout. Đây là hai lượt huấn luyện/đánh giá GPU thật, không thay `results/verdict.json` hoặc bốn run đối chứng gốc.

| Run | Replay / tổng train | Target | Regression | Format | Verdict |
|---|---:|---:|---:|---:|---|
| LoRA `correct` gốc | 0 / 225 | 0.9700 | 0.5444 | 1.0000 | FAIL |
| Replay thử 1 | 8 / 233 (3,43%) | 0.9800 | 0.7444 | 1.0000 | FAIL |
| Replay thử 2 | 11 / 236 (4,66%) | 0.9950 | 0.7444 | 1.0000 | FAIL |

Ngưỡng hồi quy là `0.7911 - 0.0200 = 0.7711`. Replay nâng điểm hồi quy thêm 0.2000 so với LoRA gốc, nhưng vẫn thấp hơn ngưỡng 0.0267; do đó không thể tuyên bố `PASSED`. Lượt 11 mẫu còn tăng target so với lượt 8 mẫu nhưng không cải thiện regression. Hai file `results/replay_3pct_verdict.json` và `results/replay_verdict.json` lưu điểm, dự đoán của 15 câu và verdict để kiểm toán. Độ trễ của replay được đo trên RTX 3060 12 GB, còn bảng NB2/NB5 gốc trên T4; không so sánh trực tiếp hai số độ trễ. Việc tiếp tục chọn replay theo đúng 15 câu hồi quy này sẽ làm bẩn holdout; cần một tập kiểm định mới trước khi thử chiến lược dữ liệu tiếp theo.

---

## 6. Định tính — ca thắng và ca thua

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt. | intent: doi_tra, urgency: cao, product: chuột không dây, sentiment: tich_cuc | Đúng format, nhưng phân loại sai trường urgency | Đúng hoàn toàn 4 trường (score 1.0) | ✅ **FT thắng**: Nhận diện chuẩn xác tính cấp thiết "Gấp" và cảm xúc tích cực dù khách yêu cầu đổi trả. |
| 2 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé. Bực mình. | intent: hoan_tien, urgency: trung_binh, product: ốp lưng điện thoại, sentiment: tieu_cuc | Sai trường sentiment (nhầm sang trung_tinh) | Đúng hoàn toàn 4 trường (score 1.0) | ✅ **FT thắng**: Bắt đúng cảm xúc "tieu_cuc" từ từ khóa "Bực mình" và ý định "hoan_tien". |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều. | intent: hoan_tien, **urgency: thap**, product: bình giữ nhiệt, sentiment: tich_cuc | Nhận diện đúng urgency: thap | Đoán sai **urgency: trung_binh** (score 0.75) | ❌ **FT thua**: Khách hỏi lịch sự "Khi nào tiện", nhưng model FT bị over-estimate mức độ cấp bách thành trung bình. |
| 4 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi. | intent: san_pham_loi, **urgency: thap**, product: nồi chiên không dầu, sentiment: trung_tinh | Nhận diện đúng urgency: thap | Đoán sai **urgency: trung_binh** (score 0.75) | ❌ **FT thua**: Tương tự ca 3, model fine-tune không nhạy với cụm từ giảm nhẹ mức độ khẩn cấp "Khi nào tiện". |
| 5 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop nhiều. | intent: san_pham_loi, **urgency: thap**, product: áo khoác gió, sentiment: tich_cuc | Đúng ngữ cảnh lịch sự | Đoán sai **urgency: trung_binh** (score 0.75) | ❌ **FT thua**: Nhầm lẫn giữa lỗi sản phẩm thông thường và khiếu nại gấp, nâng urgency lên trung_binh. |

Cả ba ca FT thua (score 0.75) có cùng lỗi: mô hình dự đoán `urgency` từ `thap` thành `trung_binh`. Với cụm "Khi nào tiện", prompt tối ưu (b) của base model gán đúng nhãn `thap`, còn bản fine-tune dường như quá nhạy với các từ khóa khiếu nại như "bị lỗi", "chưa thấy tiền", "thiếu phụ kiện".

---

## 7. Kết luận & điều tôi học được

**Kết luận.**
Nếu xét dưới góc độ một hệ thống triển khai toàn diện phục vụ đa mục đích, ta **chưa nên deploy** bản fine-tune này ngay lập tức vào production, bởi vì mô hình đã không vượt qua cổng kiểm soát hồi quy (năng lực kiến thức tổng quát sụt giảm mạnh -0.247). Tuy nhiên, nếu hệ thống được thiết kế theo kiến trúc microservices chuyên biệt — nơi model chỉ đóng vai trò một module backend biệt lập nhận văn bản ticket và trả về JSON phân loại để định tuyến yêu cầu cho nhân viên hỗ trợ — thì bản fine-tune này lại là một ứng viên xuất sắc: nó đạt độ chính xác tác vụ mục tiêu lên tới 97.0% (vượt trội hoàn toàn so với baseline prompt tối ưu 76.5%), định dạng JSON chuẩn xác 100%, và tốc độ xử lý nhanh chóng (1.4s). 

Các yếu tố đã chứng minh có tác dụng trong bài này gồm che loss mask đúng (`assistant-only`), dùng learning rate 1e-4 thay vì 1e-5, và gắn adapter lên đầy đủ lớp tuyến tính của text decoder. Thí nghiệm đối chứng ngân sách cho thấy `attn_only` rank 283 không thắng `correct` rank 16 trên target; thí nghiệm B4 bổ sung cho thấy tăng rank **ở cùng vị trí `text-linear`** lại cải thiện target trên 50 ticket. Vì vậy hiệu quả của rank phụ thuộc vị trí gắn và điều kiện đánh giá, không suy ra từ train loss đơn lẻ. Replay 3,43% và 4,66% cải thiện regression nhưng chưa đủ qua cổng; bản LoRA vẫn chưa nên triển khai cho tác vụ đa mục đích. Bước tiếp theo cần một tập kiểm định mới và chiến lược bảo toàn năng lực tổng quát được đánh giá độc lập.

**Ba điều tôi học được:**
1. **Rank cần được đọc cùng vị trí gắn adapter**: `attn_only` ($r=283$) và `correct` ($r=16$) có cùng khoảng 32,46 triệu tham số; bản attention-only có train loss thấp hơn (0.5366 vs 0.6254) nhưng chỉ hòa target. Khi giữ nguyên vị trí `text-linear`, B4 cho thấy rank 8/16/64 đạt target 0,965/0,980/1,000. Rank giúp trên tập này nhưng không thay thế việc chọn vị trí gắn hợp lý, và chưa chứng minh chất lượng hồi quy.
2. **LoRA cần thang Learning Rate riêng biệt**: Sai lầm hạ LR xuống mức 1e-5 (thang đo full-FT) trong run `wrong_lr` đã làm đóng băng quá trình học (loss kẹt ở 1.57, accuracy 0%), cho thấy LoRA bắt buộc phải dùng LR cao hơn khoảng 10 lần so với full fine-tuning để bù đắp cho việc cập nhật ma trận rank thấp.
3. **Cổng hồi quy 4 nhóm là công cụ bảo hiểm bắt buộc**: Nếu chỉ đo đạc dựa trên accuracy của tác vụ CSKH (97.0%) hoặc train loss, ta sẽ ngộ nhận rằng việc huấn luyện thành công rực rỡ. Việc đo thêm nhóm câu hỏi tổng quát (regression) đã bóc trần hiện tượng quên thảm họa (-0.247), điều mà perplexity hay accuracy cục bộ không bao giờ chỉ ra được.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Tôi sẽ xây một tập hồi quy mới, không dùng để chọn mẫu replay, rồi kiểm tra liệu cải thiện 0.2000 trên bộ hiện tại có chuyển sang tập mới hay không. Nếu vẫn hụt, tôi sẽ mở rộng dữ liệu tổng quát từ nguồn độc lập và so sánh một chiến lược giữ năng lực gốc khác với cùng ngân sách huấn luyện; mọi kết luận triển khai chỉ dựa trên đánh giá chưa được dùng để điều chỉnh mô hình.

---

## Phụ lục — thưởng đã làm

- [x] B1 NB6 merge + hot-swap: 50 câu trước/sau đều đạt target 0,9700 (Δ = 0,0000); đã nạp `correct` và `correct_replay` trên cùng base rồi hoán đổi theo request. Bằng chứng: `results/merge_check.json`, `results/hot_swap.json`.
- [x] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`): 250 ticket CNTT trường học, gồm 200 train và 50 validation; không dùng để thay điểm core.
- [x] B3 reasoning-trace collapse: hai lượt cùng base thinking và 30 steps, `valid_trace_rate` 1,00 so với 0,15; xem bảng dưới.
- [x] B4 quét rank có kiểm soát: ba lượt GPU cùng base, vị trí `text-linear`, dữ liệu và 30 optimizer steps; xem bảng dưới.
- [x] B5 HuggingFace Hub — link: https://huggingface.co/long2711/qwen3.5-cs-triage-lora

Corpus B2 được tự biên soạn và sinh năm cách diễn đạt từ 50 tình huống nền, không lấy từ ticket người dùng thật. Script tạo dữ liệu kiểm tra 250 câu không trùng nhau và không trùng nguyên văn với các tập eval đóng băng. Cùng tình huống nền không bị chia qua hai split. Đây là bằng chứng dataset và quy trình khử nhiễm; chưa có kết quả huấn luyện/eval mô hình trên corpus này, nên tôi không gán cho nó điểm target trong bảng core.

NB6 dùng adapter `correct` trên base `unsloth/Qwen3.5-4B`, chấm đầy đủ 50 câu target trước và sau khi merge; `assert` yêu cầu điểm sau không tụt quá 0,01, kết quả thực là bằng nhau 0,9700. Mô hình gộp được lưu cục bộ vì dung lượng lớn. Ở phần hot-swap, `correct` và `correct_replay` là hai adapter đã huấn luyện riêng được nạp cùng một base và được chọn lần lượt trước mỗi request; trên ticket minh họa, hai bản trả cùng nhãn. Việc hoán đổi chạy thật nhưng một đầu ra giống nhau chưa chứng minh rằng hai adapter có chất lượng tương đương trên mọi ticket; điểm replay khác biệt được báo ở §5.1.

### B3 — mất trace khi đổi loss mask

Tôi tạo 225 rationale ngắn từ **chính nhãn và nội dung tập train**, đánh dấu đây là trace tổng hợp chứ không phải giải thích do người viết. Hai lượt dùng cùng base có chế độ thinking `unsloth/Qwen3.5-4B`, cùng prompt, dữ liệu, vị trí `text-linear`, rank 16, learning rate, seed và 30 optimizer steps; chỉ đổi `MASK_MODE`. Mẫu kiểm tra mask đầu tiên có 112 token được giám sát ở `assistant-only` và 41 ở `response-only`, xác nhận hai lượt thực sự học từ phần câu trả lời khác nhau. Cùng 20 ticket đầu của target holdout được sinh greedy với `enable_thinking=True`, tối đa 220 token. Bằng chứng và mã tái chạy: `results/reasoning_trace_bonus.json`, `scripts/reasoning_trace_bonus.py`.

| Loss mask | Trace hợp lệ | Độ chính xác bốn trường trên 20 ticket |
|---|---:|---:|
| `assistant-only` | 20/20 = 1,00 | 0,85 |
| `response-only` | 3/20 = 0,15 | 0,00 |

Chat template đặt thẻ mở `<think>` trước khi sinh; phép đo dựng lại thẻ mở đó và chỉ công nhận trace nếu đầu ra có nội dung tối thiểu 10 ký tự cùng thẻ đóng `</think>`. Lượt `response-only` mất phần đóng/nội dung hợp lệ ở 17/20 ca và cũng giảm mạnh điểm target; **thí nghiệm này không chứng minh mô hình vẫn làm đúng tác vụ khi mất trace**. Nó chứng minh rằng bỏ giám sát phần rationale trong điều kiện này làm tỷ lệ trace hợp lệ giảm từ 1,00 xuống 0,15. Vì rationale được sinh theo mẫu từ nhãn train, chưa thể suy ra chất lượng suy luận trên dữ liệu tự nhiên.

### B4 — rank ở cùng vị trí gắn adapter

Ba lượt chạy trên RTX 3060 12 GB cùng base `unsloth/Qwen3.5-4B`, split train 225 mẫu, tập target đóng băng 50 mẫu, `text-linear`, learning rate 1e-4, seed và 30 optimizer steps. `alpha=2r` giữ nguyên tỉ lệ scale LoRA. Điểm target là trung bình độ chính xác bốn trường trên 50 ticket, không phải tỷ lệ ticket đúng hoàn toàn. Bằng chứng và mã tái chạy: `results/rank_sweep.json`, `scripts/rank_sweep_bonus.py`.

| Rank | Tham số trainable | Train loss | Target | VRAM đỉnh |
|---:|---:|---:|---:|---:|
| 8 | 16.232.448 | 0,4570 | 0,965 | 8,44 GB |
| 16 | 32.464.896 | 0,3518 | 0,980 | 8,68 GB |
| 64 | 129.859.584 | 0,2390 | 1,000 | 10,31 GB |

Trong thí nghiệm này, tăng rank từ 8 lên 16 và 64 cải thiện target lần lượt 0,015 và 0,020, nên rank là đòn bẩy **khi vị trí gắn adapter, dữ liệu, lịch huấn luyện và cách chấm đã giữ cố định**, và khi ngân sách bộ nhớ cho phép. Rank 64 dùng khoảng tám lần tham số của rank 8 và thêm 1,87 GB VRAM đỉnh; lợi ích được đo trên đúng 50 ticket này. Điểm 1,000 không chứng minh tổng quát hóa hoặc cổng hồi quy: ba adapter sweep chưa được chấm trên regression và không thay verdict FAILED của core. Lượt rank 16 địa phương là thí nghiệm bổ sung trên RTX 3060; điểm core rank 16 trong NB5 vẫn là 0,970 trên T4, vì vậy không trộn hai môi trường vào cùng phép so sánh.
