# Text-to-Image AI Generator (Stable Diffusion + Gradio)

Ce projet est une **application simple de génération d’images à partir de texte** (*Text-to-Image*) basée sur **Stable Diffusion 1.5** et une interface web construite avec **Gradio**.

Il permet à n’importe quel utilisateur de saisir un prompt textuel et d’obtenir une image générée automatiquement par un modèle d’intelligence artificielle.

---

##  Fonctionnalités

* Génération d’images à partir de descriptions textuelles
* Utilisation du modèle **Stable Diffusion v1.5**
* Interface web intuitive avec **Gradio**
* Fonctionne **sans GPU** (CPU uniquement, plus lent mais accessible)
* Possibilité de partager l’interface via un lien public

---

## Technologies utilisées

* **Python 3.9+**
* **PyTorch**
* **Diffusers (Hugging Face)**
* **Gradio**
* **Stable Diffusion v1.5**

---

## Installation

### 1️ Cloner le projet

```bash
git clone https://github.com/Dave-kossi/text-to-image-gradio.git
cd text-to-image-gradio
```

### 2️ Créer un environnement virtuel (recommandé)

```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
```

### 3️ Installer les dépendances

```bash
pip install torch diffusers gradio transformers accelerate
```

⚠️ **Remarque** : Le premier lancement téléchargera automatiquement le modèle Stable Diffusion (plusieurs Go).

---

## Lancer l’application

```bash
python app.py
```

Une interface Gradio s’ouvrira automatiquement dans votre navigateur.

Si `share=True` est activé, un **lien public temporaire** sera également généré.

---

##  Structure du code

```text
.
├── image.py          # Script principal
├── README.md       # Documentation du projet
```

### 🔹 Chargement du modèle

```python
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cpu")
```

* Chargement du modèle Stable Diffusion 1.5
* Exécution sur CPU (compatible avec les machines sans GPU)

### 🔹 Génération d’image

```python
def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image
```

* Prend un texte en entrée
* Retourne une image générée par le modèle

### 🔹 Interface Gradio

```python
demo = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter your prompt"),
    outputs=gr.Image(label="Generated Image"),
    title="Text-to-Image AI Generator",
    description="Generate images from text using free AI."
)
```

---

##  Exemple de prompt

> *"A futuristic city at sunset, ultra realistic, cinematic lighting"*

---

## Améliorations possibles

* Support GPU (CUDA)
* Choix du style artistique
* Paramètres avancés (steps, guidance scale, seed)
* Sauvegarde automatique des images
* Déploiement sur Hugging Face Spaces

---

## Licence

Ce projet est fourni à des fins **éducatives et expérimentales**.

Le modèle Stable Diffusion est soumis à la licence de **Hugging Face / RunwayML**.

---

## Auteur

**Kossi Noumagno**
Data Analyst / Data Scientist
Passionné par l’IA, la data science et les applications intelligentes

---

 *N’hésite pas à laisser une étoile au projet si tu l’aimes !*
