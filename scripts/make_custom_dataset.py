"""Build a separate, deterministic campus IT helpdesk corpus for bonus B2.

The original frozen Day21 data and all core scores remain untouched. Every
scenario below was authored for this corpus; style variants are synthetic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "custom_campus_helpdesk.jsonl"
INSTRUCTION = (
    "Phân loại yêu cầu hỗ trợ CNTT trong trường thành JSON với đúng bốn khóa: "
    "intent, urgency, product, sentiment. Chỉ trả về JSON. "
    "intent thuộc: cap_quyen | su_co | huong_dan | thay_doi | bao_mat; "
    "urgency thuộc: cao | trung_binh | thap; "
    "sentiment thuộc: tieu_cuc | trung_tinh | tich_cuc; "
    "product là tên dịch vụ hoặc thiết bị được nhắc tới."
)

# (Service, issue). Each row describes a distinct, plausible helpdesk request.
SCENARIOS = {
    "cap_quyen": [
        ("VLearn", "tài khoản của tôi chưa được cấp quyền xem học phần đã đăng ký"),
        ("email sinh viên", "tài khoản mới chưa có quyền truy cập hộp thư"),
        ("thư viện số", "tài khoản không mở được tài liệu điện tử đã đăng ký"),
        ("cổng đăng ký môn", "tôi không có quyền vào mục đăng ký học phần"),
        ("VPN trường", "tài khoản chưa được cấp quyền kết nối VPN từ nhà"),
        ("hệ thống đặt phòng lab", "nhóm của tôi chưa có quyền đặt phòng thực hành"),
        ("Microsoft 365", "giấy phép tài khoản chưa bật ứng dụng Office"),
        ("ứng dụng campus", "tài khoản không được cấp quyền xem thẻ sinh viên số"),
        ("kho lưu trữ nghiên cứu", "nhóm nghiên cứu chưa được cấp quyền đọc thư mục chung"),
        ("máy in thư viện", "tài khoản sinh viên không có quyền gửi lệnh in"),
    ],
    "su_co": [
        ("Wi-Fi campus", "kết nối Wi-Fi cứ ngắt sau vài phút sử dụng"),
        ("VLearn", "trang nộp bài báo lỗi khi tôi tải tệp PDF lên"),
        ("email sinh viên", "hộp thư không đồng bộ thư mới trên máy tính"),
        ("thư viện số", "trang tìm kiếm tài liệu trả lỗi máy chủ"),
        ("máy in thư viện", "máy nhận lệnh nhưng không in ra giấy"),
        ("máy trạm phòng lab", "máy trạm không nhận GPU khi mở bài thực hành"),
        ("VPN trường", "VPN bị rớt khi tôi truy cập kho học liệu"),
        ("cổng đăng ký môn", "trang chọn lớp học chỉ hiện màn hình trắng"),
        ("ứng dụng campus", "mã QR vào thư viện không hiển thị"),
        ("hệ thống đặt phòng lab", "lịch đặt phòng không lưu dù đã xác nhận"),
    ],
    "huong_dan": [
        ("VLearn", "tôi muốn biết cách đổi mật khẩu tài khoản học tập"),
        ("Wi-Fi campus", "tôi cần các bước kết nối Wi-Fi bằng tài khoản sinh viên"),
        ("VLearn", "tôi muốn biết cách kiểm tra bài tập đã nộp thành công"),
        ("thư viện số", "tôi cần hướng dẫn truy cập ebook từ ngoài trường"),
        ("VPN trường", "tôi muốn biết cách cấu hình VPN trên Windows"),
        ("máy in thư viện", "tôi cần hướng dẫn gửi tài liệu đến máy in"),
        ("hệ thống đặt phòng lab", "tôi muốn biết cách đặt phòng thực hành cho nhóm"),
        ("cổng đăng ký môn", "tôi cần hướng dẫn xem lịch học trước khi chọn lớp"),
        ("email sinh viên", "tôi muốn biết cách bật xác thực hai bước"),
        ("ứng dụng campus", "tôi cần biết cách xem dung lượng lưu trữ còn lại"),
    ],
    "thay_doi": [
        ("email sinh viên", "tôi muốn đổi địa chỉ email khôi phục của tài khoản"),
        ("kho lưu trữ nghiên cứu", "nhóm cần tăng hạn mức lưu trữ cho dự án"),
        ("hệ thống đặt phòng lab", "tôi muốn đổi khung giờ đặt phòng đã xác nhận"),
        ("máy in thư viện", "tôi cần điều chỉnh hạn mức in của học kỳ"),
        ("cổng đăng ký môn", "tôi muốn chuyển sang nhóm lớp khác trong học phần"),
        ("ứng dụng campus", "tôi cần sửa tên hiển thị bị viết sai"),
        ("email sinh viên", "tôi muốn thay điện thoại nhận mã xác thực"),
        ("thư viện số", "tôi muốn đổi lịch mượn phòng đọc đã đặt"),
        ("VLearn", "trợ giảng cần thêm vai trò biên tập học phần"),
        ("VPN trường", "tôi muốn gia hạn quyền VPN cho kỳ nghiên cứu tiếp theo"),
    ],
    "bao_mat": [
        ("email sinh viên", "tôi nhận cảnh báo đăng nhập từ thiết bị lạ"),
        ("email sinh viên", "một thư giả mạo yêu cầu tôi nhập lại mật khẩu trường"),
        ("kho lưu trữ nghiên cứu", "đường dẫn chia sẻ dữ liệu nhóm bị mở công khai"),
        ("máy trạm phòng lab", "tôi để quên laptop đang đăng nhập tài khoản ở phòng lab"),
        ("email sinh viên", "điện thoại báo yêu cầu xác thực hai bước mà tôi không thực hiện"),
        ("VLearn", "tôi thấy token đăng nhập bị dán trong notebook chia sẻ"),
        ("ứng dụng campus", "tài khoản của tôi xuất hiện hoạt động mà tôi không tạo"),
        ("ứng dụng campus", "mã QR dán ở bảng thông báo dẫn tới trang đăng nhập giả"),
        ("VPN trường", "tôi phát hiện mật khẩu VPN bị chia sẻ trong nhóm chat"),
        ("email sinh viên", "một ứng dụng lạ đã được cấp quyền đọc hộp thư"),
    ],
}

STYLES = [
    ("Chào bộ phận hỗ trợ, {issue}. Mình cần dùng dịch vụ ngay cho buổi học hôm nay, mong được xử lý gấp. Tôi khá lo lắng.", "cao", "tieu_cuc"),
    ("Mình liên hệ vì {issue}. Xin phản hồi trong một hai ngày tới. Cảm ơn bộ phận hỗ trợ.", "trung_binh", "trung_tinh"),
    ("Mình liên hệ vì {issue}. Chưa cần xử lý ngay; khi thuận tiện xin phản hồi. Cảm ơn nhiều!", "thap", "tich_cuc"),
    ("Hiện tại {issue}. Tối nay mình cần dùng dịch vụ, nên mong được ưu tiên. Việc này khá phiền.", "cao", "tieu_cuc"),
    ("Về dịch vụ này, {issue}. Phản hồi trong tuần này là được; vui lòng cho biết bước tiếp theo.", "trung_binh", "trung_tinh"),
]

SECURITY_STYLES = [
    ("Tôi báo cáo tình huống bảo mật: {issue}. Xin kiểm tra ngay để giảm rủi ro. Tôi khá lo lắng.", "cao", "tieu_cuc"),
    ("Nhờ bộ phận an ninh xem gấp: {issue}. Tôi cần biết cách bảo vệ tài khoản ngay hôm nay.", "cao", "trung_tinh"),
    ("Mình vừa phát hiện: {issue}. Mong được ưu tiên xử lý trước khi thông tin bị lộ thêm.", "cao", "tieu_cuc"),
    ("Tôi cần báo cáo khẩn việc sau: {issue}. Xin hướng dẫn bước an toàn tiếp theo càng sớm càng tốt.", "cao", "trung_tinh"),
    ("Có dấu hiệu không an toàn: {issue}. Mình rất lo và mong nhận phản hồi ngay.", "cao", "tieu_cuc"),
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.casefold())).strip()


def load_frozen_inputs() -> set[str]:
    frozen = set()
    for name in ("eval_target.jsonl", "eval_regression.jsonl", "holdout_secret.jsonl"):
        path = ROOT / "data" / name
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                for key in ("input", "question", "prompt"):
                    if isinstance(row.get(key), str):
                        frozen.add(normalize(row[key]))
    return frozen


def main() -> None:
    frozen = load_frozen_inputs()
    rows = []
    seen = set()
    for intent, scenarios in SCENARIOS.items():
        for scenario_index, (product, issue) in enumerate(scenarios):
            styles = SECURITY_STYLES if intent == "bao_mat" else STYLES
            for style, urgency, sentiment in styles:
                ticket = style.format(issue=issue)
                key = normalize(ticket)
                if key in seen or key in frozen:
                    raise ValueError("Duplicate or frozen-eval overlap found")
                seen.add(key)
                label = {"intent": intent, "urgency": urgency,
                         "product": product, "sentiment": sentiment}
                rows.append({"instruction": INSTRUCTION, "input": ticket,
                             "output": json.dumps(label, ensure_ascii=False),
                             "label": label,
                             "split": "train" if scenario_index < 8 else "validation"})
    assert len(rows) == 250 and len(seen) == 250
    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                   encoding="utf-8")
    print(f"wrote={len(rows)} unique={len(seen)} frozen_exact_overlap=0 path={OUT}")


if __name__ == "__main__":
    main()
