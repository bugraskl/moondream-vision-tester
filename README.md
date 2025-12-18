# 🌕 Moondream v2 Vision Tester

A beautiful and interactive **Streamlit** web application for testing **Moondream v2**, a powerful vision-language AI model. This tool allows you to analyze images using natural language queries, detect objects with bounding boxes, and find specific points of interest.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- **📝 Query Mode**: Ask questions about images in natural language and get detailed AI-powered answers
- **📍 Point Detection**: Automatically find and mark center points of specified objects
- **📦 Object Detection**: Detect objects with bounding boxes visualization
- **🚀 GPU Acceleration**: Automatic CUDA support for faster inference
- **💾 Model Caching**: Models are cached to prevent reloading on page refresh
- **🎨 Clean UI**: Modern, responsive interface with tabbed navigation

## 🖼️ Demo

| Feature | Description |
|---------|-------------|
| Query | Ask "What is in this image?" and get detailed descriptions |
| Point | Find cracks, holes, or any objects and mark their centers |
| Detect | Draw bounding boxes around detected objects |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended for faster inference)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/bugraskl/moondream-vision-tester.git
   cd moondream-vision-tester
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run run.py
   ```

5. **Open in browser**
   
   Navigate to `http://localhost:8501`

## 📦 Requirements

Create a `requirements.txt` file with:

```text
streamlit
transformers
torch
torchvision
pillow
accelerate
```

Or install directly:
```bash
pip install streamlit transformers torch torchvision pillow accelerate
```

## ⚙️ Configuration

### Using HuggingFace Model (Default)

By default, the application downloads the Moondream v2 model from HuggingFace:
```
vikhyatk/moondream2
```

### Using Local Model

If you have the model downloaded locally, set the environment variable:

```bash
# Windows (PowerShell)
$env:LOCAL_MODEL_PATH = "C:\path\to\moondream2"

# Windows (Command Prompt)
set LOCAL_MODEL_PATH=C:\path\to\moondream2

# Linux/macOS
export LOCAL_MODEL_PATH=/path/to/moondream2
```

Then run the application:
```bash
streamlit run run.py
```

## 🎯 Usage

1. **Upload an Image**: Use the sidebar to upload an image (JPG, JPEG, PNG, or BMP)

2. **Choose an Operation**:
   - **Query Tab**: Enter a prompt and click "Run Query" to get AI-generated answers
   - **Point Tab**: Enter what to look for and click "Run Point Detection"
   - **Detect Tab**: Enter object type and click "Run Object Detection"

3. **View Results**: Results are displayed alongside the original image

### Example Prompts

| Operation | Example Input | Description |
|-----------|---------------|-------------|
| Query | "Describe this image in detail" | Get a comprehensive description |
| Query | "What colors are present?" | Analyze colors in the image |
| Point | "crack" | Find cracks in materials |
| Point | "face" | Locate faces in photos |
| Detect | "car" | Draw boxes around cars |
| Detect | "person" | Detect people in the scene |

## 🏗️ Project Structure

```
moondream-vision-tester/
├── run.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── moondream2/         # Local model files (optional)
```

## 🔧 Technical Details

- **Model**: [Moondream v2](https://huggingface.co/vikhyatk/moondream2) by vikhyatk
- **Framework**: Streamlit for web interface
- **Backend**: PyTorch with Transformers library
- **Image Processing**: Pillow (PIL)

### Performance Tips

- Use a CUDA-compatible GPU for 10-50x faster inference
- The model uses FP16 precision on GPU, FP32 on CPU
- First load takes longer due to model download/loading
- Subsequent queries are much faster due to caching

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Moondream](https://github.com/vikhyat/moondream) - The amazing vision-language model
- [Streamlit](https://streamlit.io/) - For the beautiful web framework
- [HuggingFace](https://huggingface.co/) - For model hosting and transformers library

## 📧 Contact

If you have any questions or suggestions, feel free to open an issue or reach out!

---

Made with ❤️ and 🌕
