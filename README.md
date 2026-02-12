#  FunCap: AI That Looks at Pictures & Speaks Its Mind

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Vibe](https://img.shields.io/badge/Vibe-Chaotic%20Good-purple)

**FunCap** is a deep learning project where I taught my computer to see. Does it always understand what it's looking at? No. Does it try its absolute best to describe it? Yes.

This is a **Sequence-to-Sequence (Seq2Seq)** model that takes an image and generates a natural language caption. Sometimes it's poetic, sometimes it's technically correct, and sometimes it thinks a fire hydrant is a small child in a red jacket.

## 🧠 The "Brain" (Architecture)

I didn't just glue a dictionary to a camera. Here is the actual science:

1.  **The Eyes (Encoder):**
    * Powered by **ResNet50** (pre-trained on ImageNet).
    * *Smart Move:* I "cached" the image features (2048-dim vectors) so I didn't have to melt my GPU re-processing images every epoch.
2.  **The Mouth (Decoder):**
    * An **LSTM** (Long Short-Term Memory) network.
    * It takes the image vector and predicts the next word in a sentence until it hits the `<EOS>` (End of Sentence) token or talks for too long.

## 🛠️ Tech Stack

* **Pytorch:** The heavy lifter.
* **Flickr30k:** The training ground (30,000 images).
* **Greedy & Beam Search:** Two ways for the model to decide what to say (Fast vs. Smart).

## 🚀 How to Run (Don't let your dreams be dreams)

1.  **Clone this bad boy:**
    ```bash
    git clone [https://github.com/yourusername/FunCap.git](https://github.com/yourusername/FunCap.git)
    cd FunCap
    ```

2.  **Install the boring stuff:**
    ```bash
    pip install torch torchvision nltk pandas tqdm
    ```

3.  **Train it:**
    Run the notebook. Watch the loss go down. Feel like a wizard.

## 📸 Expectation vs. Reality

| Input Image | What FunCap Says |
| :---: | :--- |
| *[Dog running]* | "A dog runs across the grass." (Nailed it) |
| *[Salad]* | "A group of people sitting at a table." (It's trying...) |

## 🤝 Contributing

If you want to make FunCap funnier, smarter, or fix my spaghetti code, feel free to open a Pull Request!

## 📜 License

[MIT](https://choosealicense.com/licenses/mit/) - Do whatever you want with it!
