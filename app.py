from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
import os

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

    file = request.files['image']
    answer_key_str = request.form.get('answer_key', '').upper()
    fill_thresh = float(request.form.get('fill_thresh', 0.35))
    
    answer_key = list(answer_key_str)
    while len(answer_key) < 100:
        answer_key.append("-")

    in_memory_file = np.frombuffer(file.read(), np.uint8)
    img_orig = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)

    # =========================================================
    # 🌟 NEW: จัดตำแหน่งภาพด้วย ORB Feature Matching (Homography)
    # อ้างอิงจากแม่แบบ (Template) เพื่อแก้ปัญหาพรินเตอร์เว้นขอบกระดาษ
    # =========================================================
    template_path = "answer sheet.png" # <--- ชื่อไฟล์แม่แบบ
    
    if os.path.exists(template_path):
        template = cv2.imread(template_path)
        template = cv2.resize(template, (REF_W, REF_H))
        
        gray_photo = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # 1. ค้นหาจุดเด่นในภาพ (Features) ด้วยอัลกอริทึม ORB
        orb = cv2.ORB_create(5000)
        kp1, des1 = orb.detectAndCompute(gray_photo, None)
        kp2, des2 = orb.detectAndCompute(gray_template, None)

        # 2. จับคู่จุดเด่นระหว่างภาพถ่ายกับภาพแม่แบบ
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(des1, des2, None)
        matches = sorted(matches, key=lambda x: x.distance, reverse=False)

        # เลือกจุดที่ตรงกันมากที่สุด 15% แรก
        good_matches = int(len(matches) * 0.15)
        matches = matches[:good_matches]

        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        # 3. คำนวณหาเมทริกซ์การบิดเบี้ยว (Homography Matrix) และดึงภาพให้ตรง
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        img = cv2.warpPerspective(img_orig, h, (REF_W, REF_H))
    else:
        # ถ้าหาไฟล์แม่แบบไม่เจอ ให้ย่อรูปแบบธรรมดา
        print("Warning: Template 'answer sheet.png' not found.")
        img = cv2.resize(img_orig, (REF_W, REF_H))

    overlay = img.copy()

    # =========================================================
    # 🌟 OMR Pipeline ปกติ (ตอนนี้ภาพถูกทาบตรงกับแม่แบบเป๊ะแล้ว!)
    # =========================================================
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray_img)
    blurred = cv2.medianBlur(gray_eq, 5)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    results = []
    correct_count, wrong_count, blank_count = 0, 0, 0

    for col_idx in range(4):
        for row_idx in range(25):
            q_num = (col_idx * 25) + row_idx
            y_pos = Y_ROWS[row_idx]
            x_pos_list = X_GROUPS[col_idx]
            key_ans = answer_key[q_num]
            
            fill_ratios = []
            detected_choices = []
            
            for choice_idx, x_pos in enumerate(x_pos_list):
                ratio_val = get_fill_ratio(closed, x_pos, y_pos, BUBBLE_R)
                fill_ratios.append(round(ratio_val, 3))
                if ratio_val >= fill_thresh:
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