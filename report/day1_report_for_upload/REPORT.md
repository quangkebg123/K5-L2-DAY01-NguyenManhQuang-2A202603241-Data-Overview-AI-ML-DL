# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> \*\*Cách dùng mẫu này:\*\* copy/điền trên máy, lưu thành `REPORT.md`, rồi \*\*upload\*\* vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1\_lab\_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp \*\*link repo\*\* trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

|Từ|Nghĩa ngắn|
|-|-|
|Prediction|Kết quả model vừa đoán ra; chưa phải nhãn đúng|
|Ground truth|Nhãn do người gán theo quy tắc|
|Taxonomy|Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80)|
|Checkpoint|File model `.pt` đã huấn luyện sẵn|
|Guideline|Quy tắc gán nhãn do người viết|
|Annotator|Người gán nhãn|
|Reviewer / QC|Người kiểm tra nhãn|
|Escalation|Ca khó → hỏi Lab Coach, không tự bịa|
|Model score|Số “tin” của model; không phải điểm chất lượng nhãn người|

## 1\. Phân loại ảnh

**Mở file:** `day1\_lab\_outputs/classification\_predictions.json` và `visuals/classification\_top5.png`.  
Chỉ dùng ảnh `sample\_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class\_id`= ;468 `class\_name`="cab" ; `rank`=1 ; `score`=0.510915 ; `taxonomy\_name`="ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record này nói về toàn bộ bức ảnh. Vì  "task": "image\_classification" dòng này cho cho ta biết record này phân loại hình ảnh
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách tên lớp (class list) không phải do model tự nghĩ ra. Nó được xác định bởi bộ dữ liệu huấn luyện (taxonomy) và thường được lưu hoặc liên kết với model checkpoint.
4. Vì sao cần ghi cả `class\_id`, `class\_name` và `taxonomy\_name`, không chỉ ghi mỗi tên lớp?
Cần ghi cả class\_id, class\_name và taxonomy\_name vì ba trường này giải quyết ba vấn đề khác nhau: định danh lớp, giúp con người đọc hiểu và xác định hệ thống phân loại đang sử dụng
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Khi trong một ảnh có nhiều vật thể hoặc nhiều chủ đề, Guideline cần quy định rõ tiêu chí chọn nhãn đại diện cho toàn bộ ảnh (Primary Label) để những người gán nhãn khác nhau đưa ra kết quả thống nhất.
6. Vì sao không được lấy `score` của model làm ground truth?
Không được lấy score của model làm Ground Truth (nhãn chuẩn) vì score chỉ thể hiện mức độ tự tin của model đối với dự đoán, chứ không chứng minh dự đoán đó đúng

## 2\. Phát hiện vật thể

**Mở file:** `day1\_lab\_outputs/detection\_predictions.json` và `visuals/detection\_predictions.png`.  
Chỉ dùng ảnh `sample\_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class\_name`="person" ; `score`=0.912625 ; `bbox\_xyxy`=\[385.33, 69.24,498.92, 348.92] ; `bbox\_width`=113.58 ; `bbox\_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
 Góc trên-trái (x1, y1): (385.33, 69.24),Góc dưới-phải (x2, y2): (498.92, 348.92),Chiều rộng: 113.58 pixel,Chiều cao: 279.68 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
ngưỡng 0.2 có 1 prediction vì 0.912625 > 0.2 . ngưỡng 0.6 có 1 prediction vì  0.912625 > 0.6
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Hạ ngưỡng → tăng số prediction được giữ lại → reviewer phải kiểm tra nhiều hộp hơn → tăng thời gian QC và nguy cơ False Positive.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Người gán nhãn phải vẽ một bounding box hình chữ nhật sát nhất có thể với toàn bộ phần nhìn thấy của từng đối tượng cần gán nhãn.
Các yêu cầu thực hiện:
- Mỗi đối tượng riêng biệt có một bounding box riêng.
- Bao phủ đầy đủ phần đối tượng nhìn thấy, không bỏ sót bộ phận.
- Không lấy thêm vùng nền hoặc vật thể bên cạnh vào bounding box.
- Nếu đối tượng bị che khuất, chỉ vẽ quanh phần nhìn thấy, không tự suy đoán phần bị che
- Nếu đối tượng bị cắt bởi mép ảnh, box được phép chạm mép ảnh.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Nếu đây là tình huống phổ biến và cách xử lý đã rõ ràng thì tôi viết thành quy tắc guideline. Khi không xác định được vật thể có đủ điều kiện để gán nhãn hay không , không rõ phần bị     che có được tính là một đối tượng độc lập không và Guideline hiện tại chưa quy định cách xử lý thì nên hỏi Lab Coach

## 3\. Phân đoạn theo từng đối tượng

**Mở file:** `day1\_lab\_outputs/segmentation\_predictions.json` và `visuals/segmentation\_prediction.png`.  
Chỉ dùng ảnh `sample\_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance\_id`="kitchen-008"; `class\_name`="spoon" ; `score`=0.487577 ; số điểm polygon=`polygon\_point\_count`=46; 3–5 điểm đầu của `polygon\_xy`=\[\[544.0, 67.0], \\\[544.0, 80.0], \[545.0, 81.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Polygon thể hiện đường viền và hình dạng chi tiết của vật thể, giúp xác định chính xác vùng vật thể hơn  bounding box hình chữ nhật, đồng thời hạn chế lấy nhầm vùng nền
3. `instance\_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance\_id dùng để phân biệt 1 class riêng biệt. Ví dụ 1 class bowl trong ảnh có nhiều bowl để xác định từng bowl riêng biệt ra sử dụng instance\_id
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Biên mask phải bám sát đường viền thực tế cảu vật thể, không lấy thêm vùng nền, không bỏ sot phần nhìn thấy và tách riêng các vật thể khi chúng chạm nhau
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
góc trái dưỡi bowl và dinner tabled chạm nhau và bị che. Tôi sẽ áp dụng guideline để vẽ mask theo phần nhìn thấy, không thể xác định chính xác và guideline không có quy định xử lý tôi sẽ báo cáo Lab Coach (escakation) để đc hướng dẫn, tránh gán nhãn sai.

## 4\. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

|Tác vụ|Ground truth trông như thế nào? (đơn vị nhãn)|Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON|Annotator làm gì?|Reviewer xem gì?|
|-|-|-|-|-|
|Phân loại ảnh|Mỗi ảnh có một nhãn lớp ví dụ :dog,cat.Json lưu class hoặc label|Ảnh mờ, có nhiều đối tượng dễ phân biệt nhầm|Quan sát hhh và gán nhẫn đúng theo guideline|Kiểm tra nhãn có đung nội dug hhh và quy tắc không|
|Phát hiện vật thể|Mỗi vật thể có nhãn và hộp giới hạn (bounding box)|Box lệch thiếu vật thể,box quá rộng hoặc qua hẹp|Ve bõ bao quanh từng vật thể và gán |Kiểm tra vị trí , kích thước box, số lượng vã nhãn vật thể|
|Instance segmentation|Mỗi vật thể có nhãn,ID riêng và vùng mask/polygon thể hiện hình dạng|Mask bị lem thiếu vùng dinh hai vật thể hoặc sai đường viền|Vẽ mask theo đường viền từng vật thể, tách riêng các instance|Kiểm tra độ chính xac đường viền  vùng mask, ID à nhãn|

## 5\. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ sử dụng dữ liệu đúng mục đích bài lab, không chia sẻ thông tin nhạy cảm và không công khai dữ liệu khi chưa đc phép
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Tôi sẽ dừng ngay và báo cáo cho giảng viên hoặc người pụ trách baig lab để được hướng dẫn

## 6\. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

* \[ ] `REPORT.md` (file này đã điền)
* \[ ] `day1\_lab\_outputs/classification\_predictions.json`
* \[ ] `day1\_lab\_outputs/detection\_predictions.json`
* \[ ] `day1\_lab\_outputs/segmentation\_predictions.json`
* \[ ] `day1\_lab\_outputs/IMAGE\_ATTRIBUTION.md`
* \[ ] `day1\_lab\_outputs/visuals/classification\_top5.png`
* \[ ] `day1\_lab\_outputs/visuals/detection\_predictions.png`
* \[ ] `day1\_lab\_outputs/visuals/segmentation\_prediction.png`
* \[ ] Ô validation cuối notebook báo `PASS`
* \[ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
* \[ ] Đã nộp **link repository** trên LMS

