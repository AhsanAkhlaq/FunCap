#  FunCap: AI That Looks at Pictures & Speaks Its Mind

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Vibe](https://img.shields.io/badge/Vibe-Immaculate-purple)

**FunCap** is a deep learning-powered web application that generates natural language descriptions for any image you upload. It uses a **ResNet50** to see and a **GRU-based Decoder** to speak. 

Simply upload a photo, and watch the AI try its best to describe reality. (Sometimes it's poetic, sometimes it thinks a toaster is a radio. That's the fun part.)


## 🧠 Under the Hood

The architecture has been streamlined for performance:

1.  **The Eyes (Encoder):** * **ResNet50** (Pre-trained on ImageNet). We strip off the last layer and use it to extract rich 2048-dimensional feature vectors.
2.  **The Brain (Decoder):** * **GRU (Gated Recurrent Unit):** A streamlined RNN that takes the image features and generates sentences word-by-word. 
    * *Why GRU?* It's faster and often just as effective as LSTM for this task.
3.  **The Strategy (Beam Search):**
    * Instead of just picking the "most likely" next word (Greedy), we explore the top 5 possible sentence paths simultaneously to generate the most coherent caption possible.

## 📂 Project Structure

* `app.py`: The main Streamlit web application.
* `tokenizer.json`: The vocabulary file (you need this to translate numbers back to English).
* `caption_model.pth`: The trained PyTorch model weights.
* `requirements.txt`: List of dependencies.


## 🚀 How to Run (Don't let your dreams be dreams)

1.  **Clone this bad boy:**
    ```bash
    git clone [https://github.com/AhsanAkhlaq/FunCap.git](https://github.com/AhsanAkhlaq/FunCap.git)
    cd FunCap
    ```

2.  **Install the some stuff:**
    ```bash
    pip install torch pytorch streamlit tokenizers pillow 
    ```

3.  **Launch the App**
      Run the following command in your terminal:
     ```bash
    streamlit run app.py
     ```
   A browser window should pop up automatically at http://localhost:8501.

## 📸 Usage
   1. Click "Browse files" and select a .jpg or .png.
   2. Wait for the image to appear.
   3. Click the "Generate Caption" button.
   4. Judge the AI's creativity.

## 📸 Examples

<img width="753" height="651" alt="image" src="https://github.com/user-attachments/assets/0c713bf6-0dbe-4259-b6f5-27ef51c16a2a" />
<img width="571" height="680" alt="image" src="https://github.com/user-attachments/assets/1f0d8636-3553-49c8-b918-d94accace68f" />
<img width="502" height="730" alt="image" src="https://github.com/user-attachments/assets/a428b002-e410-4c65-bbcc-f27a488ef9a0" />




## 🤝 Contributing

If you want to make FunCap funnier, smarter, or fix my spaghetti code, feel free to open a Pull Request!

## 📜 License

[MIT](https://choosealicense.com/licenses/mit/) - Do whatever you want with it!
