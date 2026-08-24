import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import numpy as np
import cv2
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import io
import qrcode

# 1. Page Config
st.set_page_config(page_title="Advanced AI Studio", layout="wide")
st.title("🚀 Advanced AI & Computer Vision Studio")

# 2. Sidebar Navigation
st.sidebar.header("⚙️ Select AI Feature")
mode = st.sidebar.selectbox("Choose Module:", [
    "Image Processing & Filters", 
    "Face & Eye Detection (AI)", 
    "Color Palette Extractor",
    "QR Code Scanner & Generator",
    "Watermark Creator",
    "Live Webcam Studio", 
    "Image Resizer & Converter", 
    "ML Data Simulator"
])

# ----------------- Module 1: Image Filters -----------------
if mode == "Image Processing & Filters":
    st.subheader("📷 Real-Time Image Editing & Filters")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
            
        st.sidebar.subheader("🎛️ Adjust Controls")
        brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0)
        contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0)
        blur = st.sidebar.slider("Blur", 0, 10, 0)
        filter_type = st.sidebar.selectbox("Filter:", ["None", "Grayscale", "Edge Detection", "Contour", "Invert Color"])
        
        processed = ImageEnhance.Brightness(image.copy()).enhance(brightness)
        processed = ImageEnhance.Contrast(processed).enhance(contrast)
        if blur > 0:
            processed = processed.filter(ImageFilter.GaussianBlur(blur))
            
        img_arr = np.array(processed)
        if filter_type == "Grayscale":
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        elif filter_type == "Edge Detection":
            img_arr = cv2.Canny(img_arr, 100, 200)
        elif filter_type == "Contour":
            img_arr = cv2.Laplacian(cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY), cv2.CV_8U)
        elif filter_type == "Invert Color":
            img_arr = cv2.bitwise_not(img_arr)
            
        with col2:
            st.image(img_arr, caption=f"Result ({filter_type})", use_container_width=True)
            res_pil = Image.fromarray(img_arr)
            buf = io.BytesIO()
            res_pil.save(buf, format="PNG")
            st.download_button("📥 Download Result", buf.getvalue(), "processed.png", "image/png")

# ----------------- Module 2: Face Detection -----------------
elif mode == "Face & Eye Detection (AI)":
    st.subheader("👤 AI Face & Eye Detector")
    uploaded_file = st.file_uploader("Upload Human Portrait Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_arr = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img_arr, (x, y), (x + w, y + h), (50, 205, 50), 4)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img_arr[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                    
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            with col2:
                st.image(img_arr, caption=f"Detected Faces: {len(faces)}", use_container_width=True)
                st.success(f"Total faces detected: {len(faces)}")
        except Exception as e:
            st.error("Error loading face detection classifier.")

# ----------------- Module 3: Color Palette Extractor -----------------
elif mode == "Color Palette Extractor":
    st.subheader("🎨 Dominant Color Palette Generator")
    uploaded_file = st.file_uploader("Upload Image for Color Analysis", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        num_colors = st.sidebar.slider("Number of Colors", 2, 8, 5)
        
        img_resized = image.resize((150, 150))
        img_np = np.array(img_resized).reshape(-1, 3)
        
        kmeans = KMeans(n_clusters=num_colors, random_state=42)
        kmeans.fit(img_np)
        colors = kmeans.cluster_centers_.astype(int)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.write("### Extracted Dominant Colors (HEX):")
            for color in colors:
                hex_code = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                st.color_picker(f"Color: {hex_code}", hex_code, key=hex_code)

# ----------------- Module 4: QR Code Scanner & Generator -----------------
elif mode == "QR Code Scanner & Generator":
    st.subheader("📱 QR Code Generator & Scanner")
    tab1, tab2 = st.tabs(["Generate QR Code", "Scan QR Code"])
    
    with tab1:
        text_input = st.text_input("Enter URL or Text to generate QR Code:", "https://github.com")
        if text_input:
            qr = qrcode.make(text_input)
            buf = io.BytesIO()
            qr.save(buf, format="PNG")
            st.image(buf.getvalue(), caption="Generated QR Code", width=250)
            st.download_button("📥 Download QR Code", buf.getvalue(), "qrcode.png", "image/png")
            
    with tab2:
        qr_file = st.file_uploader("Upload QR Code Image", type=["jpg", "png", "jpeg"])
        if qr_file:
            img = Image.open(qr_file)
            img_np = np.array(img.convert("RGB"))
            detector = cv2.QRCodeDetector()
            data, bbox, _ = detector.detectAndDecode(img_np)
            if data:
                st.success(f"Decoded Data: **{data}**")
            else:
                st.warning("No valid QR code found in the image.")

# ----------------- Module 5: Watermark Creator -----------------
elif mode == "Watermark Creator":
    st.subheader("🏷️ Add Text Watermark to Image")
    uploaded_file = st.file_uploader("Upload Image for Watermarking", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGBA")
        wm_text = st.sidebar.text_input("Watermark Text:", "CONFIDENTIAL")
        text_size = st.sidebar.slider("Text Size", 20, 100, 40)
        
        txt_img = Image.new("RGBA", image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt_img)
        
        x = image.width - (len(wm_text) * (text_size // 2)) - 20
        y = image.height - text_size - 20
        draw.text((x, y), wm_text, fill=(255, 255, 255, 180))
        
        watermarked = Image.alpha_composite(image, txt_img).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(watermarked, caption="Watermarked Result", use_container_width=True)
            
            buf = io.BytesIO()
            watermarked.save(buf, format="PNG")
            st.download_button("📥 Download Watermarked Image", buf.getvalue(), "watermark.png", "image/png")

# ----------------- Module 6: Live Webcam -----------------
elif mode == "Live Webcam Studio":
    st.subheader("📸 Live Camera Input")
    cam_file = st.camera_input("Take Picture")
    if cam_file:
        st.image(cam_file, caption="Webcam Photo Captured!")

# ----------------- Module 7: Converter -----------------
elif mode == "Image Resizer & Converter":
    st.subheader("📐 Image Resizer & Quality Compressor")
    uploaded_file = st.file_uploader("Upload Image to Compress", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Original Size: {image.size}", width=300)
        
        new_width = st.slider("Width", 100, 2000, image.width)
        new_height = st.slider("Height", 100, 2000, image.height)
        
        resized_img = image.resize((new_width, new_height))
        st.image(resized_img, caption=f"New Size: ({new_width}, {new_height})", width=300)

# ----------------- Module 8: ML Simulator -----------------
elif mode == "ML Data Simulator":
    st.subheader("📈 Interactive K-Means Cluster Visualizer")
    k = st.sidebar.slider("Clusters (K)", 2, 8, 3)
    pts = st.sidebar.slider("Data Points", 50, 1000, 300)
    
    X, _ = make_blobs(n_samples=pts, centers=k, random_state=42)
    st.scatter_chart(pd.DataFrame(X, columns=['X', 'Y']))
