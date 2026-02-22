"""
Real-Time ISL Word Detection with Sentence Formation
======================================================
Uses webcam to detect Indian Sign Language words and form sentences.

Controls:
- SPACE: Add current word to sentence
- BACKSPACE: Remove last word from sentence
- C: Clear entire sentence
- R: Reset current detection
- S: Save screenshot
- Q: Quit
"""

import cv2
import torch
import numpy as np
import os
from PIL import Image
import torchvision.transforms as T
from word_model import WordCNN
from collections import deque
import time

# ══════════════════════════════════════════════════════════════
# PATHS (relative to script location)
# ══════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
# Use the balanced 30-word dataset
DATASET_PATH = os.path.join(PROJECT_DIR, "data_Set")
MODEL_PATH = os.path.join(PROJECT_DIR, "word_model_30classes.pth")
CONFIDENCE_THRESHOLD = 0.35       # Minimum confidence to show prediction (lowered for better detection)
SMOOTHING_WINDOW = 12             # Number of frames for majority voting
DISPLAY_TOP_K = 3                 # Show top K predictions
AUTO_ADD_THRESHOLD = 0.50         # Auto-add word if confidence exceeds this
AUTO_ADD_FRAMES = 25              # Frames to hold for auto-add
MAX_SENTENCE_WORDS = 15           # Maximum words in sentence

# ══════════════════════════════════════════════════════════════
# DEVICE SETUP
# ══════════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ══════════════════════════════════════════════════════════════
# LOAD MODEL & WORD LABELS
# ══════════════════════════════════════════════════════════════
if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Handle both checkpoint formats (dict with metadata or just state_dict)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format with metadata
        word_labels = checkpoint['class_labels']
        num_classes = checkpoint['num_classes']
        print(f"✓ Model trained accuracy: {checkpoint.get('val_acc', 'N/A')}%")
    else:
        # Old format (just state_dict) - load labels from dataset folder
        word_labels = sorted(os.listdir(DATASET_PATH))
        num_classes = len(word_labels)
        checkpoint = {'model_state_dict': checkpoint}
    
    word_map = {i: w for i, w in enumerate(word_labels)}
    print(f"✓ Loaded {num_classes} word classes: {word_labels}")
    
    model = WordCNN(num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Model ready!")
else:
    print(f"⚠ Model file '{MODEL_PATH}' not found!")
    print("  Please run 'python train_balanced_dataset.py' first to train the model.")
    exit(1)

# ══════════════════════════════════════════════════════════════
# IMAGE TRANSFORM
# ══════════════════════════════════════════════════════════════
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ══════════════════════════════════════════════════════════════
# PREDICTION SMOOTHING & SENTENCE STATE
# ══════════════════════════════════════════════════════════════
prediction_history = deque(maxlen=SMOOTHING_WINDOW)
confidence_history = deque(maxlen=SMOOTHING_WINDOW)

# Sentence building
sentence_words = []
last_added_word = None
stable_word_counter = 0
last_stable_word = None

def get_smoothed_prediction():
    """Returns majority-voted prediction and average confidence."""
    if len(prediction_history) == 0:
        return None, 0.0
    
    from collections import Counter
    counter = Counter(prediction_history)
    most_common = counter.most_common(1)[0]
    pred_label = most_common[0]
    vote_ratio = most_common[1] / len(prediction_history)
    
    relevant_confs = [c for p, c in zip(prediction_history, confidence_history) if p == pred_label]
    avg_conf = sum(relevant_confs) / len(relevant_confs) if relevant_confs else 0
    
    return pred_label, avg_conf * vote_ratio

# Display name mapping for cleaner output
DISPLAY_NAMES = {
    "I_ME_MINE_MY": "I",
    "HELLO_HI": "Hello",
}

def get_display_name(word):
    """Get clean display name for a word."""
    if word in DISPLAY_NAMES:
        return DISPLAY_NAMES[word]
    # Default: remove underscores and title case
    return word.replace("_", " ").title()

def add_word_to_sentence(word):
    """Add a word to the sentence."""
    global last_added_word, stable_word_counter
    
    if len(sentence_words) >= MAX_SENTENCE_WORDS:
        return False
    
    # Use clean display name
    clean_word = get_display_name(word)
    
    sentence_words.append(clean_word)
    last_added_word = word
    stable_word_counter = 0
    
    # Clear prediction history after adding
    prediction_history.clear()
    confidence_history.clear()
    
    return True

def get_sentence_text():
    """Get the current sentence as a string."""
    if not sentence_words:
        return ""
    return " ".join(sentence_words)

# ══════════════════════════════════════════════════════════════
# WEBCAM INITIALIZATION
# ══════════════════════════════════════════════════════════════
print("Initializing webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("✗ Error: Could not open webcam!")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("✓ Webcam ready")

# ══════════════════════════════════════════════════════════════
# UI COLORS (BGR format)
# ══════════════════════════════════════════════════════════════
COLOR_BG = (30, 30, 30)
COLOR_PRIMARY = (0, 230, 118)      # Green
COLOR_SECONDARY = (255, 191, 0)    # Cyan
COLOR_WARNING = (0, 165, 255)      # Orange
COLOR_ACCENT = (147, 112, 219)     # Purple
COLOR_TEXT = (255, 255, 255)
COLOR_MUTED = (150, 150, 150)
COLOR_SENTENCE_BG = (40, 40, 50)

# ══════════════════════════════════════════════════════════════
# DRAW UI FUNCTIONS
# ══════════════════════════════════════════════════════════════
def draw_confidence_bar(img, x, y, width, height, confidence, color):
    """Draw a confidence bar."""
    cv2.rectangle(img, (x, y), (x + width, y + height), (60, 60, 60), -1)
    fill_width = int(width * confidence)
    cv2.rectangle(img, (x, y), (x + fill_width, y + height), color, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (100, 100, 100), 1)

def draw_progress_ring(img, center, radius, progress, color, thickness=3):
    """Draw a circular progress indicator."""
    # Background circle
    cv2.circle(img, center, radius, (60, 60, 60), thickness)
    # Progress arc
    if progress > 0:
        end_angle = int(360 * progress)
        cv2.ellipse(img, center, (radius, radius), -90, 0, end_angle, color, thickness)

# ══════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  REAL-TIME ISL WORD DETECTION + SENTENCE BUILDER")
print("="*60)
print("  A / SPACE: Accept - Add word to sentence")
print("  D        : Reject - Delete wrong prediction")
print("  BACKSPACE: Undo - Remove last word")
print("  C        : Clear entire sentence")
print("  R        : Reset detection")
print("  S        : Save screenshot")
print("  Q        : Quit")
print("="*60 + "\n")

frame_count = 0
fps_start = time.time()
fps = 0
word_added_flash = 0  # Flash effect when word is added

with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]
        
        # ──────────────────────────────────────────
        # INFERENCE
        # ──────────────────────────────────────────
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        
        top_k_probs, top_k_indices = torch.topk(probabilities, DISPLAY_TOP_K)
        top_k_probs = top_k_probs.cpu().numpy()
        top_k_indices = top_k_indices.cpu().numpy()
        
        best_conf = top_k_probs[0]
        best_idx = top_k_indices[0]
        
        if best_conf > CONFIDENCE_THRESHOLD:
            prediction_history.append(best_idx)
            confidence_history.append(best_conf)
        
        smoothed_pred, smoothed_conf = get_smoothed_prediction()
        current_word = word_map[smoothed_pred] if smoothed_pred is not None else None
        
        # ──────────────────────────────────────────
        # AUTO-ADD LOGIC (hold sign to add)
        # ──────────────────────────────────────────
        if current_word and smoothed_conf > AUTO_ADD_THRESHOLD:
            if current_word == last_stable_word:
                stable_word_counter += 1
            else:
                stable_word_counter = 1
                last_stable_word = current_word
            
            # Auto-add after holding
            if stable_word_counter >= AUTO_ADD_FRAMES and current_word != last_added_word:
                if add_word_to_sentence(current_word):
                    word_added_flash = 15
                    print(f"✓ Added: {get_display_name(current_word)}")
        else:
            stable_word_counter = 0
            last_stable_word = None
        
        # ──────────────────────────────────────────
        # FPS CALCULATION
        # ──────────────────────────────────────────
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start)
            fps_start = time.time()
        
        # Flash effect decay
        if word_added_flash > 0:
            word_added_flash -= 1
        
        # ──────────────────────────────────────────
        # DRAW UI - TOP PANEL
        # ──────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame_width, 140), COLOR_BG, -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
        
        # Title
        cv2.putText(frame, "ISL SENTENCE BUILDER", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_PRIMARY, 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame_width - 120, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_MUTED, 1)
        
        # ──────────────────────────────────────────
        # CURRENT WORD DETECTION
        # ──────────────────────────────────────────
        if smoothed_pred is not None and smoothed_conf > 0.25:
            word = word_map[smoothed_pred]
            display_word = get_display_name(word)  # Clean display name
            
            cv2.putText(frame, "CURRENT SIGN:", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MUTED, 1)
            
            # Color based on confidence
            if smoothed_conf > 0.7:
                color = COLOR_PRIMARY
            elif smoothed_conf > 0.5:
                color = COLOR_SECONDARY
            else:
                color = COLOR_WARNING
            
            # Flash effect when word added
            if word_added_flash > 0:
                color = COLOR_ACCENT
            
            cv2.putText(frame, display_word.upper(), (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # Confidence bar
            draw_confidence_bar(frame, 20, 115, 180, 15, smoothed_conf, color)
            cv2.putText(frame, f"{smoothed_conf*100:.0f}%", (210, 127),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_MUTED, 1)
            
            # Auto-add progress ring
            if stable_word_counter > 0 and current_word != last_added_word:
                progress = min(stable_word_counter / AUTO_ADD_FRAMES, 1.0)
                ring_center = (280, 90)
                draw_progress_ring(frame, ring_center, 20, progress, COLOR_ACCENT, 3)
                cv2.putText(frame, "HOLD", (260, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_MUTED, 1)
            
            # Show Accept/Reject options
            cv2.putText(frame, "[A] Accept", (320, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_PRIMARY, 1)
            cv2.putText(frame, "[D] Reject", (320, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)
        else:
            cv2.putText(frame, "Show a sign...", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_MUTED, 2)
        
        # ──────────────────────────────────────────
        # TOP K PREDICTIONS (Right side)
        # ──────────────────────────────────────────
        panel_x = frame_width - 250
        cv2.putText(frame, "Predictions:", (panel_x, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MUTED, 1)
        
        for i, (prob, idx) in enumerate(zip(top_k_probs, top_k_indices)):
            word = word_map[idx]
            y_pos = 60 + i * 25
            
            display_word = word[:12] + ".." if len(word) > 12 else word
            cv2.putText(frame, display_word, (panel_x, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT, 1)
            
            bar_color = COLOR_PRIMARY if i == 0 else COLOR_MUTED
            draw_confidence_bar(frame, panel_x + 100, y_pos - 10, 60, 12, prob, bar_color)
            cv2.putText(frame, f"{prob*100:.0f}%", (panel_x + 170, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_MUTED, 1)
        
        # ──────────────────────────────────────────
        # SENTENCE DISPLAY (Bottom panel)
        # ──────────────────────────────────────────
        sentence_panel_height = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, frame_height - sentence_panel_height), 
                     (frame_width, frame_height), COLOR_SENTENCE_BG, -1)
        frame = cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)
        
        # Sentence label
        cv2.putText(frame, "SENTENCE:", (20, frame_height - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ACCENT, 1)
        
        # Sentence text
        sentence_text = get_sentence_text()
        if sentence_text:
            # Word count
            cv2.putText(frame, f"({len(sentence_words)}/{MAX_SENTENCE_WORDS} words)", 
                        (120, frame_height - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_MUTED, 1)
            
            # Truncate if too long for display
            max_display_len = 70
            if len(sentence_text) > max_display_len:
                sentence_text = "..." + sentence_text[-(max_display_len-3):]
            
            cv2.putText(frame, sentence_text, (20, frame_height - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
        else:
            cv2.putText(frame, "Start signing to build a sentence...", 
                        (20, frame_height - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_MUTED, 1)
        
        # Controls hint
        cv2.putText(frame, "[A] Accept  [D] Reject  [BS] Undo  [C] Clear  [Q] Quit", 
                    (frame_width - 420, frame_height - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_MUTED, 1)
        
        # ──────────────────────────────────────────
        # DISPLAY
        # ──────────────────────────────────────────
        cv2.imshow("ISL Sentence Builder", frame)
        
        # ──────────────────────────────────────────
        # KEY HANDLING
        # ──────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\n" + "="*50)
            print("FINAL SENTENCE:", get_sentence_text() if sentence_words else "(empty)")
            print("="*50)
            break
        
        elif key == ord(' ') or key == ord('a') or key == ord('A'):  # SPACE or A - Accept/Add word
            if current_word:
                if add_word_to_sentence(current_word):
                    word_added_flash = 15
                    print(f"✓ Added: {get_display_name(current_word)}")
        
        elif key == 8 or key == 127:  # BACKSPACE - Remove last word
            if sentence_words:
                removed = sentence_words.pop()
                last_added_word = sentence_words[-1] if sentence_words else None
                print(f"✗ Removed: {removed}")
        
        elif key == ord('c') or key == ord('C'):  # Clear sentence
            sentence_words.clear()
            last_added_word = None
            print("Sentence cleared!")
        
        elif key == ord('r') or key == ord('R'):  # Reset detection
            prediction_history.clear()
            confidence_history.clear()
            stable_word_counter = 0
            last_stable_word = None
            print("Detection reset!")
        
        elif key == ord('d') or key == ord('D'):  # Reject/Delete current prediction
            if current_word:
                print(f"✗ Rejected: {current_word}")
                # Clear prediction history to ignore this sign
                prediction_history.clear()
                confidence_history.clear()
                stable_word_counter = 0
                last_stable_word = None
                # Prevent this word from being auto-added immediately
                last_added_word = current_word
        
        elif key == ord('s') or key == ord('S'):  # Screenshot
            filename = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")

# ══════════════════════════════════════════════════════════════
# CLEANUP
# ══════════════════════════════════════════════════════════════
cap.release()
cv2.destroyAllWindows()
print("Application closed.")
