# app.py
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import io
import time

# Configuration de la page
st.set_page_config(
    page_title="Générateur d'Images IA",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache pour le modèle (ne se recharge pas à chaque interaction)
@st.cache_resource
def load_model():
    """Charge le modèle Stable Diffusion une seule fois"""
    st.info("🔄 Chargement du modèle IA... Cette opération peut prendre quelques minutes.")
    
    model_id = "runwayml/stable-diffusion-v1-5"
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        pipe = pipe.to("cpu")
        st.success("✅ Modèle chargé avec succès!")
        return pipe
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {e}")
        return None

def generate_image(pipe, prompt):
    """Génère une image à partir du prompt"""
    try:
        with st.spinner("🖼️ Création de votre image..."):
            start_time = time.time()
            
            # Génération de l'image
            image = pipe(
                prompt,
                num_inference_steps=50,
                guidance_scale=7.5
            ).images[0]
            
            generation_time = time.time() - start_time
            st.info(f"⏱️ Temps de génération: {generation_time:.1f} secondes")
            
        return image
    except Exception as e:
        st.error(f"❌ Erreur lors de la génération: {e}")
        return None

def main():
    # En-tête de l'application
    st.title("🎨 Générateur d'Images IA")
    st.markdown("""
    Créez des images étonnantes à partir de descriptions textuelles grâce à l'IA.
    Utilisez **Stable Diffusion** gratuitement!
    """)
    
    # Sidebar pour les paramètres
    with st.sidebar:
        st.header("⚙️ Paramètres")
        
        st.subheader("Instructions")
        st.markdown("""
        1. Entrez votre description en français ou anglais
        2. Cliquez sur **Générer l'Image**
        3. Téléchargez votre création!
        """)
        
        st.subheader("Exemples de prompts")
        example_prompts = [
            "Un chat astronaut dans l'espace, style cartoon",
            "Paysage montagneux avec un lac cristallin au coucher du soleil",
            "Ville futuriste avec des voitures volantes, style cyberpunk",
            "Dragon jouant du piano dans une forêt enchantée",
            "Intérieur cosy d'un café avec des livres et des plantes"
        ]
        
        for example in example_prompts:
            if st.button(example, key=example):
                st.session_state.prompt = example
    
    # Chargement du modèle
    pipe = load_model()
    
    if pipe is None:
        st.error("Impossible de charger le modèle. Vérifiez votre connexion internet.")
        return
    
    # Zone principale
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Description de l'image")
        
        # Zone de texte pour le prompt
        prompt = st.text_area(
            "Décrivez l'image que vous souhaitez créer:",
            height=100,
            placeholder="Ex: Un chat astronaut explorant Mars avec un drapeau...",
            key="prompt_input",
            value=st.session_state.get('prompt', '')
        )
        
        # Paramètres avancés
        with st.expander("🔧 Paramètres avancés"):
            num_steps = st.slider(
                "Nombre d'étapes de génération",
                min_value=20,
                max_value=100,
                value=50,
                help="Plus d'étapes = meilleure qualité mais plus lent"
            )
            
            guidance_scale = st.slider(
                "Guidance Scale",
                min_value=1.0,
                max_value=20.0,
                value=7.5,
                help="Contrôle combien l'image suit le prompt"
            )
        
        # Bouton de génération
        generate_btn = st.button(
            "🚀 Générer l'Image", 
            type="primary",
            disabled=not prompt,
            use_container_width=True
        )
    
    with col2:
        st.subheader("🖼️ Image Générée")
        
        # Affichage des résultats
        if generate_btn and prompt:
            # Génération de l'image
            image = generate_image(pipe, prompt)
            
            if image:
                # Affichage de l'image
                st.image(image, use_column_width=True, caption="Votre image générée")
                
                # Téléchargement de l'image
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                
                st.download_button(
                    label="📥 Télécharger l'image",
                    data=buf.getvalue(),
                    file_name=f"image_ia_{hash(prompt)}.png",
                    mime="image/png",
                    use_container_width=True
                )
                
                # Affichage du prompt utilisé
                st.text_area("Prompt utilisé:", prompt, height=80)
        
        elif 'generated_image' in st.session_state:
            # Affiche l'image précédente si elle existe
            st.image(st.session_state.generated_image, use_column_width=True)
        else:
            # Message d'attente
            st.info("👆 Entrez une description et cliquez sur 'Générer l'Image' pour commencer!")
            
            # Image de placeholder
            st.image("https://via.placeholder.com/512x512/4B5563/FFFFFF?text=Image+à+générer", 
                    use_column_width=True, 
                    caption="Votre image apparaîtra ici")
    
    # Section d'exemples en bas
    st.markdown("---")
    st.subheader("💡 Idées de création")
    
    examples_cols = st.columns(5)
    example_images = [
        ("🏔️", "Paysage alpin avec chalet"),
        ("🐉", "Dragon dans un château médiéval"),
        ("🚀", "Fusée décollant au coucher du soleil"),
        ("🏙️", "Métropole futuriste de nuit"),
        ("🌊", "Océan avec baleines lumineuses")
    ]
    
    for i, (emoji, desc) in enumerate(example_images):
        with examples_cols[i]:
            if st.button(f"{emoji}\n{desc}", use_container_width=True):
                st.session_state.prompt = desc
                st.rerun()

# Gestion des erreurs globales
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Une erreur est survenue: {e}")
        st.info("🔧 Essayez de rafraîchir la page ou de réessayer plus tard.")