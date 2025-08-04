# 🎈 SquintyPics - IllusionDiffusion Streamlit App

A kid-safe image remixer powered by IllusionDiffusion AI. Transform any image into magical optical illusions!

## ✨ Features

- **Kid-friendly interface** with curated, safe prompts
- **Automatic image loading** from watch folder  
- **Two creative themes**: Imaginative & Realistic
- **Print & email** generated artwork
- **Hidden admin panel** for customization
- **Optical illusion generation** using ControlNet

## 🚀 Quick Start

### Windows
```bash
# Run the setup script
setup.bat

# Or manually:
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Linux/Mac
```bash
# Run the setup script
./setup.sh

# Or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 📁 File Structure

```
squintypics/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── input_images/       # Drop images here for processing
├── prompts.json       # Generated config file
├── IllusionDiffusion/ # Original IllusionDiffusion code
└── setup.bat/.sh      # Setup scripts
```

## 🎮 How to Use

1. **Add images**: Drop PNG/JPG files into `input_images/` folder
2. **Open app**: Visit http://localhost:8501 in your browser
3. **Choose theme**: Select "Imaginative" or "Realistic"
4. **Pick prompt**: Choose from curated kid-safe prompts
5. **Generate**: Click "✨ Remix!" to create illusion art
6. **Share**: Print or email your creations

## 🔧 Admin Features

Access admin panel with password `letmein` to:
- Edit prompt categories and options
- Change watch folder location
- Configure email settings
- Customize app behavior

## 📋 Requirements

- **Python 3.11+** (recommended for best compatibility)
- **8GB+ RAM** (for model loading)
- **GPU recommended** (CUDA support for faster generation)
- **10GB+ disk space** (for downloaded models)

## ⚡ Performance Notes

- **First run**: Downloads ~6GB of AI models (takes 10-30 minutes)
- **Generation time**: 30-60 seconds per image
- **GPU acceleration**: Automatically detected if available
- **Memory usage**: ~4-6GB during generation

## 🛠️ Troubleshooting

**Models won't load?**
- Ensure you have enough RAM and disk space
- Check internet connection for model downloads
- Try restarting the app

**Import errors?**
- Verify Python 3.11 is being used
- Reinstall requirements: `pip install -r requirements.txt`

**Slow generation?**
- GPU acceleration improves speed significantly
- Reduce image size in watch folder
- Close other memory-intensive applications

## 🔒 Safety

- All prompts are curated for family-friendly content
- No inappropriate content generation
- Admin controls for customization
- Local processing (no data sent to external servers)

---

Built with ❤️ using [IllusionDiffusion](https://huggingface.co/spaces/AP123/IllusionDiffusion), Streamlit, and Stable Diffusion