"""
반응하는 우포포 — STEP 1 스타터 코드
웹캠 → MediaPipe(얼굴·손 인식) → OSC 송신 (TouchDesigner 수신용)

실행 전 준비 (STEP 0):
    Python 3.10 또는 3.11 가상환경에서
    pip install mediapipe opencv-python python-osc

실행:
    python upopo_tracker.py
    → 웹캠 창이 뜨고, 콘솔에 송신 값이 출력됨
    → TouchDesigner에서 OSC In CHOP (포트 7000)으로 수신

보내는 OSC 주소:
    /face/size  : 얼굴 크기 (0.0~1.0, 클수록 가까움) — R1 접근 감지용
    /face/x     : 얼굴 중심 x좌표 (0.0~1.0, 0=왼쪽) — R2 시선 추적용
    /hand/wave  : 손 흔들기 감지 (0 또는 1)          — R3 화답용

종료: 웹캠 창에서 q
"""

import time
from collections import deque

import cv2
import mediapipe as mp
from pythonosc.udp_client import SimpleUDPClient

# ── 설정 ─────────────────────────────────────────────
OSC_IP = "127.0.0.1"     # TouchDesigner가 같은 컴퓨터면 그대로
OSC_PORT = 7000          # TD의 OSC In CHOP 포트와 맞출 것
CAM_INDEX = 0            # 웹캠이 여러 개면 1, 2로 바꿔보기
WAVE_WINDOW_SEC = 1.0    # 손 흔들기 판정: 최근 1초간의 손목 움직임 관찰
WAVE_MIN_SWINGS = 3      # 1초 안에 좌우 방향 전환이 3회 이상이면 '흔들기'
# ─────────────────────────────────────────────────────

osc = SimpleUDPClient(OSC_IP, OSC_PORT)

mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)
mp_hands = mp.solutions.hands.Hands(
    max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# 손목 x좌표 이력 (타임스탬프, x값) — 흔들기 판정용
wrist_history = deque()


def detect_wave(now: float) -> int:
    """최근 1초간 손목 x좌표의 좌우 방향 전환 횟수로 흔들기 판정."""
    # 오래된 기록 제거
    while wrist_history and now - wrist_history[0][0] > WAVE_WINDOW_SEC:
        wrist_history.popleft()
    if len(wrist_history) < 5:
        return 0
    xs = [x for _, x in wrist_history]
    # 방향 전환 횟수 세기
    swings = 0
    prev_dir = 0
    for i in range(1, len(xs)):
        diff = xs[i] - xs[i - 1]
        if abs(diff) < 0.01:          # 미세 떨림 무시
            continue
        direction = 1 if diff > 0 else -1
        if prev_dir != 0 and direction != prev_dir:
            swings += 1
        prev_dir = direction
    return 1 if swings >= WAVE_MIN_SWINGS else 0


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없음 — CAM_INDEX를 바꿔볼 것")

    print(f"OSC 송신 시작 → {OSC_IP}:{OSC_PORT}  (종료: q)")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)                     # 거울 모드
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # MediaPipe는 RGB
        now = time.time()

        face_size = 0.0
        face_x = 0.5

        # ── 얼굴 ──
        face_result = mp_face.process(rgb)
        if face_result.detections:
            det = face_result.detections[0]
            box = det.location_data.relative_bounding_box
            face_size = max(0.0, min(1.0, box.width))          # 폭 = 거리 근사
            face_x = max(0.0, min(1.0, box.xmin + box.width / 2))
            # 시각화
            h, w = frame.shape[:2]
            p1 = (int(box.xmin * w), int(box.ymin * h))
            p2 = (int((box.xmin + box.width) * w), int((box.ymin + box.height) * h))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

        # ── 손 ──
        hand_result = mp_hands.process(rgb)
        if hand_result.multi_hand_landmarks:
            wrist = hand_result.multi_hand_landmarks[0].landmark[0]  # 0 = 손목
            wrist_history.append((now, wrist.x))
            h, w = frame.shape[:2]
            cv2.circle(frame, (int(wrist.x * w), int(wrist.y * h)), 8, (255, 0, 255), -1)

        wave = detect_wave(now)

        # ── OSC 송신 ──
        osc.send_message("/face/size", face_size)
        osc.send_message("/face/x", face_x)
        osc.send_message("/hand/wave", wave)

        # 화면 표시
        cv2.putText(frame, f"size:{face_size:.2f} x:{face_x:.2f} wave:{wave}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("upopo tracker  (q = quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
