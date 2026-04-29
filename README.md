# Tanglish Fake Review Detection

This project builds a stronger fake-review detector for Tanglish (`Tamil + English`) reviews using a hybrid approach:

- word TF-IDF
- character TF-IDF
- handcrafted deception cues
- optional transformer upgrade (`MuRIL` or `XLM-R`)

## Files

- `tanglish_2000_reviews.csv`: main Tanglish dataset
- `fake reviews dataset.csv`: auxiliary English dataset
- `train_hybrid.py`: hybrid baseline trainer
- `requirements.txt`: Python dependencies
- `PROJECT_GUIDE.md`: project explanation and modeling strategy

## Run order

1. Install Python 3.11 or newer
2. Create a virtual environment
3. Install dependencies
4. Place both CSV files in this folder
5. Run the baseline:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_hybrid.py
```

## Next upgrade

After the hybrid baseline works:

- extract transformer embeddings from `google/muril-base-cased` or `xlm-roberta-base`
- combine them with the hybrid features
- compare with the baseline using macro F1
