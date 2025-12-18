import streamlit as st
from transformers import AutoModelForCausalLM
import torch
from PIL import Image, ImageDraw
import os

# -----------------------------------------------------------------------------
# SETTINGS & MODEL LOADING
# -----------------------------------------------------------------------------

# Model configuration
# You can set LOCAL_MODEL_PATH environment variable to use a local model
# Otherwise, it will download from HuggingFace
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", None)
HUGGINGFACE_MODEL_ID = "vikhyatk/moondream2"

# Page Settings
st.set_page_config(page_title="Moondream v2 Tester", layout="wide")
st.title("🌕 Moondream v2 - Visual Analysis Test Interface")

@st.cache_resource
def load_moondream_model():
    """
    Loads the model once and caches it.
    Prevents reloading on page refresh.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"⚙️ Loading model... (**Device:** {device})")
    
    try:
        # Use local model if path is set and exists, otherwise use HuggingFace
        if LOCAL_MODEL_PATH and os.path.exists(LOCAL_MODEL_PATH):
            model_path = LOCAL_MODEL_PATH
            use_local = True
            st.info(f"📁 Using local model from: {model_path}")
        else:
            model_path = HUGGINGFACE_MODEL_ID
            use_local = False
            st.info(f"🌐 Downloading model from HuggingFace: {model_path}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
            local_files_only=use_local
        )
        st.success(f"✅ Model loaded successfully: {model_path}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_moondream_model()

# -----------------------------------------------------------------------------
# INTERFACE LOGIC
# -----------------------------------------------------------------------------

# Sidebar - Image Upload
st.sidebar.header("1. Upload Image")
uploaded_file = st.sidebar.file_uploader("Select an image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file and model:
    # Convert image to PIL format
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Results & Operations")
        
        # Tabbed interface: Query, Point, Detect
        tab1, tab2, tab3 = st.tabs(["📝 Query (Q&A)", "📍 Point", "📦 Detect (Bounding Box)"])

        # --- TAB 1: QUERY ---
        with tab1:
            st.info("Ask questions about the image or request descriptions.")
            prompt = st.text_area("Prompt:", value="Describe this image.", height=100)
            
            if st.button("Run Query", type="primary"):
                with st.spinner("Model is thinking..."):
                    enc_image = model.encode_image(image)
                    answer = model.query(
                        enc_image,
                        prompt,
                        settings={"temperature": 0.0, "top_p": 0.1, "max_tokens": 64},
                        reasoning=True,
                    )
                    st.write("### Answer:")
                    st.markdown(f"> {answer}")

        # --- TAB 2: POINT ---
        with tab2:
            st.info("Find center points of objects in the image.")
            point_label = st.text_input("What to mark? (e.g., car, truck, bus)", value="car")
            
            if st.button("Run Point Detection"):
                with st.spinner("Searching for points..."):
                    enc_image = model.encode_image(image)
                    points = model.point(enc_image, point_label)
                    
                    # Visualization
                    draw_image = image.copy()
                    draw = ImageDraw.Draw(draw_image)
                    width, height = draw_image.size
                    
                    # Data format can be list or dict, check it
                    pts_list = points["points"] if isinstance(points, dict) else points
                    
                    st.write(f"Number of points found: {len(pts_list)}")
                    
                    for p in pts_list:
                        x = p['x'] * width
                        y = p['y'] * height
                        # Draw red dot
                        r = 5
                        draw.ellipse((x-r, y-r, x+r, y+r), fill=(255, 0, 0, 255))
                    
                    st.image(draw_image, caption=f"Point Result: {point_label}", use_container_width=True)

        # --- TAB 3: DETECT ---
        with tab3:
            st.info("Detect objects with bounding boxes.")
            detect_label = st.text_input("What to detect?", value="car")
            
            if st.button("Run Object Detection"):
                with st.spinner("Detecting objects..."):
                    # Detect method typically takes PIL image directly
                    detect_result = model.detect(image, detect_label)
                    
                    draw_image = image.copy()
                    draw = ImageDraw.Draw(draw_image)
                    width, height = draw_image.size
                    
                    objects = detect_result["objects"] if isinstance(detect_result, dict) else detect_result
                    st.write(f"Number of objects found: {len(objects)}")
                    
                    for obj in objects:
                        x_min = obj['x_min'] * width
                        y_min = obj['y_min'] * height
                        x_max = obj['x_max'] * width
                        y_max = obj['y_max'] * height
                        
                        # Draw red bounding box
                        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
                    
                    st.image(draw_image, caption=f"Detection Result: {detect_label}", use_container_width=True)

else:
    if not uploaded_file:
        st.info("👈 Please upload an image from the sidebar to start testing.")