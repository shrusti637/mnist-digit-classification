import streamlit as st
import joblib
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.ndimage import center_of_mass, shift as scipy_shift

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Page config
st.set_page_config(page_title="Handwritten Digit Classifier", layout="wide", page_icon="✍️")

# Custom CSS for "Hyper-Rich" Futuristic UI
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
    /* Animated Gradient Background */
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .stApp {
        background: linear-gradient(-45deg, #020617, #0f172a, #131313, #020617);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        color: #f8fafc;
        font-family: 'Outfit', sans-serif;
    }

    /* Glass HUD Container */
    .hud-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 4px; /* Sharp futuristic edges */
        padding: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
        margin-bottom: 2rem;
    }

    /* Neon Corner Brackets for HUD */
    .hud-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 15px; height: 15px;
        border-top: 3px solid #10b981;
        border-left: 3px solid #10b981;
    }
    .hud-card::after {
        content: '';
        position: absolute;
        bottom: 0; right: 0;
        width: 15px; height: 15px;
        border-bottom: 3px solid #3b82f6;
        border-right: 3px solid #3b82f6;
    }

    .hud-card:hover {
        border-color: rgba(16, 185, 129, 0.5);
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.2);
        transform: translateY(-2px);
        transition: 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    /* Sidebar Depth */
    [data-testid="stSidebar"] {
        background: rgba(2, 6, 23, 0.95) !important;
        border-right: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    /* Futuristic Prediction Value */
    .hud-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 160px;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(16, 185, 129, 0.5);
        line-height: 1;
        margin: 0;
    }

    /* Glowing Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #10b981, #059669) !important;
        border: none !important;
        color: white !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        padding: 1rem !important;
        border-radius: 4px !important;
        transition: all 0.3s !important;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.3) !important;
    }
    .stButton>button:hover {
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.6) !important;
        transform: scale(1.02);
    }

    /* Headings */
    h1, h2, h3 {
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #f1f5f9;
        border-left: 4px solid #10b981;
        padding-left: 15px;
    }

    /* Big Bold Metric Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: #10b981;
        background: rgba(16, 185, 129, 0.05);
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.1);
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #94a3b8;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        background: linear-gradient(135deg, #f8fafc, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-accent {
        color: #10b981;
    }

    /* Hide standard footer */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Helper for Data Augmentation (Jittering)
def get_jittered_images(image_data):
    """Generates 5 jittered versions of the input image (1x784)."""
    img_2d = image_data.reshape(28, 28)
    jitters = []
    # Original
    jitters.append(image_data)
    # Shifts: (y, x)
    offsets = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1)]
    for dy, dx in offsets:
        shifted = scipy_shift(img_2d, shift=(dy, dx), mode='constant', cval=0.0)
        jitters.append(shifted.reshape(1, -1))
    return np.vstack(jitters)

# Load Model
@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "mnist_model.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

if 'model' not in st.session_state:
    st.session_state.model = load_model()

# --- REFACTORED STATE MANAGEMENT ---
state_defaults = {
    'last_processed_img': None,
    'prediction': None,
    'probs': None,
    'prediction_made': False,
    'feedback_received': False,
    'show_correction_hub': False
}
for key, val in state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

model = st.session_state.model

# Sidebar Navigation
st.sidebar.title("🔢 MNIST System")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Draw", "Dashboard", "Analytics"])

if page == "Draw":
    st.title("✍️ Draw Handwritten Digit")
    st.write("Draw a digit (0-9) inside the black box. Click 'Predict Digit' to see the magic!")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Interactive Canvas")
        
        # Brush Control
        with st.expander("🖌️ BRUSH SETTINGS", expanded=False):
            stroke_width = st.slider("Brush Size", 5, 50, 20)
            stroke_color = st.color_picker("Brush Color", "#FFFFFF")
            
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color="#000000",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        # Reset prediction state if the user starts a new drawing
        if canvas_result.json_data is not None:
            current_canvas_str = str(canvas_result.json_data.get("objects", []))
            if 'last_canvas_str' not in st.session_state:
                st.session_state.last_canvas_str = current_canvas_str
            elif st.session_state.last_canvas_str != current_canvas_str:
                st.session_state.last_canvas_str = current_canvas_str
                st.session_state.prediction_made = False
                st.session_state.feedback_received = False
                st.session_state.show_correction_hub = False

        if st.button("Clear Canvas"):
            st.session_state.prediction_made = False
            st.session_state.feedback_received = False
            st.session_state.show_correction_hub = False
            st.rerun()

    with col2:
        st.subheader("Prediction Result")
        if model is None:
            st.warning("Model not found. Please run the training script.")
        else:
            if canvas_result.image_data is not None:
                # 1. Convert to grayscale and threshold
                img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
                img_np = np.array(img)
                
                # 2. Find bounding box of the digit
                rows = np.any(img_np > 50, axis=1) # Threshold of 50
                cols = np.any(img_np > 50, axis=0)
                
                if not np.any(rows) or not np.any(cols):
                    st.warning("Please draw something on the canvas first!")
                    prediction = None
                else:
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    
                    # 3. Crop to the digit (inclusive of the last detected pixel)
                    crop = img.crop((cmin, rmin, cmax + 1, rmax + 1))
                    
                    # 4. Resize with aspect ratio preservation (fitting into 20x20)
                    w, h = crop.size
                    ratio = 20.0 / max(w, h)
                    new_size = (int(w * ratio), int(h * ratio))
                    crop = crop.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # 5. Paste onto a 28x28 black background using Center of Mass
                    temp_img = Image.new('L', (28, 28), 0)
                    upper = (28 - new_size[1]) // 2
                    left = (28 - new_size[0]) // 2
                    temp_img.paste(crop, (left, upper))
                    
                    com = center_of_mass(np.array(temp_img))
                    shift_y = 14 - com[0]
                    shift_x = 14 - com[1]
                    
                    final_img = Image.new('L', (28, 28), 0)
                    final_img.paste(crop, (int(left + shift_x), int(upper + shift_y)))
                    
                    img_data = np.array(final_img).reshape(1, -1) / 255.0
                    
                    if st.button("🚀 INITIATE NEURAL PREDICTION", key="predict_btn"):
                        st.session_state.last_processed_img = img_data
                        st.session_state.prediction = model.predict(img_data)[0]
                        st.session_state.probs = model.predict_proba(img_data)[0]
                        st.session_state.prediction_made = True
                        st.session_state.feedback_received = False
                        st.session_state.show_correction_hub = False

                    # --- STABLE UI RENDERING ---
                    if st.session_state.prediction_made:
                        prediction = st.session_state.prediction
                        probs = st.session_state.probs
                        
                        st.markdown(f"""
                        <div class="hud-card">
                            <h3 style="border:none; padding:0; margin-bottom:10px; font-size: 0.8rem; color: #10b981; opacity: 0.6; letter-spacing:3px;">[ CLASSIFICATION RESULT ]</h3>
                            <div class="hud-value">{prediction}</div>
                            <div style="text-align: center; margin-top: 1rem;">
                                <div style="display:inline-block; border-left: 3px solid #10b981; padding-left: 15px; text-align: left;">
                                    <span style="color: #64748b; font-size: 0.7rem; display: block; text-transform: uppercase; letter-spacing:1px;">Confidence Level</span>
                                    <span style="color: #10b981; font-size: 1.8rem; font-weight: 700; font-family: 'JetBrains Mono';">{probs[int(prediction)]*100:.2f}%</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # --- STEP 1: VERIFICATION HUD ---
                        if not st.session_state.feedback_received:
                            st.subheader("🧐 Is this prediction correct?")
                            v_col1, v_col2 = st.columns(2)
                            with v_col1:
                                if st.button("✅ YES, EXCELLENT", use_container_width=True):
                                    st.session_state.feedback_received = True
                                    st.balloons()
                                    st.success("Great! System calibration confirmed.")
                            with v_col2:
                                if st.button("❌ NO, INCORRECT", use_container_width=True):
                                    st.session_state.feedback_received = True
                                    st.session_state.show_correction_hub = True

                        # --- STEP 2: BOOSTED CORRECTION HUB ---
                        if st.session_state.show_correction_hub:
                            st.markdown("---")
                            st.subheader("🛠️ Boosted Neural Correction Hub")
                            st.write("The system needs training. Specify the correct digit and initiate a high-intensity re-alignment cycle.")
                            
                            c1, c2 = st.columns([2, 1])
                            with c1:
                                correct_digit = st.selectbox("What was the actual digit?", range(10), key="correct_digit_box")
                            with c2:
                                st.write("") # Spacer
                                st.write("") # Spacer
                                if st.button("💾 BOOSTED WEIGHT UPDATE"):
                                    with st.spinner("⚡ Initiating High-Intensity Neural Re-alignment..."):
                                        try:
                                            # 1. Data Augmentation (Jittering)
                                            augmented_batch = get_jittered_images(st.session_state.last_processed_img)
                                            labels_batch = np.array([str(correct_digit)] * len(augmented_batch))
                                            
                                            # Load replay buffer to prevent Catastrophic Forgetting
                                            try:
                                                replay_buffer_path = os.path.join(BASE_DIR, "replay_buffer.joblib")
                                                replay_buffer = joblib.load(replay_buffer_path)
                                                X_replay = replay_buffer['X']
                                                y_replay = replay_buffer['y']
                                                
                                                # Over-sample the new drawing to give it significance
                                                X_combined = np.vstack([X_replay, np.repeat(augmented_batch, 5, axis=0)])
                                                y_combined = np.concatenate([y_replay, np.repeat(labels_batch, 5)])
                                            except FileNotFoundError:
                                                X_combined = augmented_batch
                                                y_combined = labels_batch
                                                
                                            # 2. Iterative Training (reduced iterations, combined dataset)
                                            for _ in range(3):
                                                st.session_state.model.partial_fit(X_combined, y_combined)
                                            
                                            # 3. Persist
                                            model_path = os.path.join(BASE_DIR, "mnist_model.joblib")
                                            joblib.dump(st.session_state.model, model_path)
                                            
                                            # 4. Live Flip - Update session state so UI reflects it immediately
                                            st.session_state.prediction = st.session_state.model.predict(st.session_state.last_processed_img)[0]
                                            st.session_state.probs = st.session_state.model.predict_proba(st.session_state.last_processed_img)[0]
                                            st.session_state.show_correction_hub = False # Close hub after success
                                            
                                            st.success(f"System successfully retrained! Calibration for '{correct_digit}' is complete.")
                                            st.rerun() # Refresh to show the flipped prediction
                                        except Exception as e:
                                            st.error(f"Neural training interrupted: {str(e)}")

                        # Visual Neural Buffer Section
                        st.subheader("🤖 Visual Neural Buffer")
                        st.image(final_img, width=150)
                        
                        # Pixel Insight
                        with st.expander("🛠️ DECRYPT DATA MATRIX"):
                            st.write("Numerical input stream analysis:")
                            fig_px, ax_px = plt.subplots(figsize=(8, 8))
                            fig_px.patch.set_facecolor('#050505')
                            sns.heatmap(np.array(final_img), annot=False, cmap='magma', 
                                        cbar=False, ax=ax_px, xticklabels=False, yticklabels=False)
                            st.pyplot(fig_px)
                    
                        # Probability Distribution
                        st.subheader("📊 Signal Distribution")
                        fig, ax = plt.subplots(figsize=(6, 3))
                        fig.patch.set_facecolor('#050505')
                        ax.set_facecolor('#050505')
                        ax.bar(range(10), probs, color='#10b981', alpha=0.8)
                        ax.set_xticks(range(10))
                        ax.set_xticklabels(range(10), color='#94a3b8')
                        ax.set_yticks([]) 
                        for spine in ax.spines.values(): spine.set_visible(False)
                        st.pyplot(fig)

elif page == "Dashboard":
    st.title("📊 Model Dashboard")
    
    metrics_path = os.path.join(BASE_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Test Accuracy</div>
                <div class="metric-value"><span class="metric-accent">{metrics['accuracy']:.2f}%</span></div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Categorical Loss</div>
                <div class="metric-value">{metrics['test_loss']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Training metrics not found. Training might be in progress...")

    st.divider()
    st.subheader("Sample Predictions from Test Set")
    
    samples_path = os.path.join(BASE_DIR, "samples.json")
    if os.path.exists(samples_path):
        with open(samples_path, "r") as f:
            samples = json.load(f)
        
        cols = st.columns(5)
        for i, sample in enumerate(samples[:10]):
            with cols[i % 5]:
                img = np.array(sample['image'])
                st.image(img, width=120, caption=f"True: {sample['target']} | Pred: {sample['pred']}")
    else:
        st.info("Sample predictions will appear after training.")

elif page == "Analytics":
    st.title("📈 Advanced Analytics")
    
    st.subheader("Confusion Matrix")
    cm_path = os.path.join(BASE_DIR, "confusion_matrix.npy")
    if os.path.exists(cm_path):
        cm = np.load(cm_path)
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor('#0e1117')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax, 
                    annot_kws={"size": 12}, cbar=False)
        ax.set_xlabel('Predicted labels', color='white', fontsize=14)
        ax.set_ylabel('True labels', color='white', fontsize=14)
        ax.set_xticklabels(range(10), color='white')
        ax.set_yticklabels(range(10), color='white')
        ax.set_title("Confusion Matrix", color='white', fontsize=18)
        st.pyplot(fig)
        
        st.markdown("""
        ### 🔍 Insights
        - **Precision:** The model shows high sensitivity across all digits.
        - **Common Misclassifications:** Look for larger numbers off the diagonal. For example, '9' and '4' or '3' and '5' are often confused due to visual similarity.
        - **Accuracy:** The current MLP architecture reaches optimized performance on grayscale handwritten data.
        """, unsafe_allow_html=True)
    else:
        st.warning("Analytics data not found. Please run the training script.")
