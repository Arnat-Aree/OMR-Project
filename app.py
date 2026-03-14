from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64

app = Flask(__name__)

# คอนฟิกกระดาษ
REF_W, REF_H = 759, 1000
BUBBLE_R = 11
CHOICES = list("ABCDE")
X_GROUPS = [
    [57,  89, 116, 143, 170],
    [241, 268, 296, 323, 350],
    [417, 449, 476, 503, 530],
    [598, 629, 656, 683, 711],
]
Y_ROWS = [203, 236, 268, 301, 333, 365, 398, 430, 463, 495, 529, 561, 593, 626, 658, 691, 723, 755, 788, 821, 854, 886, 919, 951, 983]

def get_fill_ratio(binary_img, cx, cy, r):
    mask = np.zeros_like(binary_img)
    cv2.circle(mask, (cx, cy), r-2, 255, -1)
    masked_data = cv2.bitwise_and(binary_img, binary_img, mask=mask)
    total_pixels = cv2.countNonZero(mask)
    return cv2.countNonZero(masked_data) / total_pixels if total_pixels > 0 else 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_omr():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # รับค่าพารามิเตอร์จากหน้าเว็บ
    file = request.files['image']
    answer_key_str = request.form.get('answer_key', '').upper()
    fill_thresh = float(request.form.get('fill_thresh', 0.35))
    
    # เตรียมเฉลยให้ครบ 100 ข้อ
    answer_key = list(answer_key_str)
    while len(answer_key) < 100:
        answer_key.append("-")

    # อ่านรูปภาพจาก memory
    in_memory_file = np.frombuffer(file.read(), np.uint8)
    img_color = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)
    img = cv2.resize(img_color, (REF_W, REF_H))
    overlay = img.copy()

    # --- Classical Image Processing Pipeline ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray)
    blurred = cv2.medianBlur(gray_eq, 5)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    results = []
    correct_count, wrong_count, blank_count = 0, 0, 0

    # --- Grading ---
    for col_idx in range(4):
        for row_idx in range(25):
            q_num = (col_idx * 25) + row_idx
            y_pos = Y_ROWS[row_idx]
            x_pos_list = X_GROUPS[col_idx]
            key_ans = answer_key[q_num]
            
            fill_ratios = []
            detected_choices = []
            
            for choice_idx, x_pos in enumerate(x_pos_list):
                ratio = get_fill_ratio(closed, x_pos, y_pos, BUBBLE_R)
                fill_ratios.append(round(ratio, 3))
                if ratio >= fill_thresh:
                    detected_choices.append(CHOICES[choice_idx])
            
            ans_str = "".join(detected_choices)
            if len(detected_choices) == 0:
                status, cls = "ไม่ตอบ", "blank"
                blank_count += 1
            elif len(detected_choices) > 1:
                status, cls = "ฝนเกิน", "multi"
                wrong_count += 1
            elif detected_choices[0] == key_ans:
                status, cls = "ถูก", "ok"
                correct_count += 1
            else:
                status, cls = "ผิด", "err"
                wrong_count += 1
                
            results.append({
                "q": q_num + 1, "ans": ans_str, "key": key_ans, 
                "status": status, "cls": cls, "ratios": fill_ratios
            })
            
            # วาด Overlay
            for choice_idx, x_pos in enumerate(x_pos_list):
                char = CHOICES[choice_idx]
                is_filled = char in detected_choices
                is_key = (char == key_ans)
                
                if is_filled and status == "ถูก":
                    cv2.circle(overlay, (x_pos, y_pos), BUBBLE_R + 2, (0, 200, 0), 2)
                elif is_filled and status != "ถูก":
                    cv2.circle(overlay, (x_pos, y_pos), BUBBLE_R + 2, (0, 0, 255), 2)
                elif is_key and not is_filled:
                    cv2.circle(overlay, (x_pos, y_pos), BUBBLE_R + 2, (0, 165, 255), 2)

    # แปลงรูปภาพที่วาดเสร็จแล้วกลับเป็น Base64 เพื่อส่งให้หน้าเว็บแสดงผล
    _, buffer = cv2.imencode('.jpg', overlay)
    overlay_b64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "score": {
            "correct": correct_count, "wrong": wrong_count, 
            "blank": blank_count, "total": 100
        },
        "results": results,
        "overlay_image": f"data:image/jpeg;base64,{overlay_b64}"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)