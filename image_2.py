# app_gradio.py
import torch
from diffusers import StableDiffusionPipeline
import gradio as gr
import time
import os
from PIL import Image

# Configuration du modèle
def load_model():
    """Charge le modèle Stable Diffusion"""
    print("🔄 Chargement du modèle Stable Diffusion...")
    
    model_id = "runwayml/stable-diffusion-v1-5"
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        pipe = pipe.to("cpu")
        print("✅ Modèle chargé avec succès!")
        return pipe
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

# Chargement initial du modèle
pipe = load_model()

def generate_image(prompt, num_steps=50, guidance_scale=7.5):
    """Génère une image à partir du prompt"""
    if not prompt:
        return None, "❌ Veuillez entrer une description"
    
    if pipe is None:
        return None, "❌ Modèle non chargé - Réessayez plus tard"
    
    try:
        start_time = time.time()
        
        # Génération de l'image
        with torch.no_grad():
            image = pipe(
                prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=512,
                width=512
            ).images[0]
        
        generation_time = time.time() - start_time
        
        success_message = f"""
        ✅ Image générée avec succès!
        ⏱️ Temps: {generation_time:.1f}s
        🎯 Étapes: {num_steps}
        """
        
        return image, success_message
        
    except Exception as e:
        error_message = f"❌ Erreur: {str(e)}"
        return None, error_message

def save_image(image, prompt):
    """Sauvegarde l'image générée"""
    if image is None:
        return None
    
    # Créer le dossier de sauvegarde
    os.makedirs("generated_images", exist_ok=True)
    
    # Nom de fichier basé sur le prompt et timestamp
    timestamp = int(time.time())
    filename = f"generated_images/image_{timestamp}.png"
    
    # Sauvegarder l'image
    image.save(filename)
    
    return filename

# Exemples de prompts
example_prompts = [
    "Un chat astronaut dans l'espace, style cartoon",
    "Paysage montagneux avec un lac cristallin au coucher du soleil", 
    "Ville futuriste avec des voitures volantes, style cyberpunk",
    "Dragon jouant du piano dans une forêt enchantée",
    "Intérieur cosy d'un café avec des livres et des plantes",
    "Robot jardinier dans une serre futuriste",
    "Forêt magique avec des champignons lumineux la nuit"
]

# Interface Gradio améliorée
with gr.Blocks(
    title="Générateur d'Images IA",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    .example-prompt {
        cursor: pointer;
        padding: 8px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .example-prompt:hover {
        background: #f0f0f0;
    }
    """
) as demo:
    
    # En-tête
    gr.Markdown("""
    # 🎨 Générateur d'Images IA
    **Créez des images étonnantes à partir de texte avec Stable Diffusion**
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Section des paramètres
            gr.Markdown("### ⚙️ Paramètres")
            
            prompt_input = gr.Textbox(
                label="Description de l'image",
                placeholder="Ex: Un chat astronaut explorant Mars avec un drapeau...",
                lines=3,
                max_lines=5,
                elem_id="prompt-input"
            )
            
            with gr.Accordion("🔧 Paramètres avancés", open=False):
                num_steps = gr.Slider(
                    minimum=20,
                    maximum=100,
                    value=50,
                    step=5,
                    label="Nombre d'étapes de génération",
                    info="Plus d'étapes = meilleure qualité mais plus lent"
                )
                
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=20.0,
                    value=7.5,
                    step=0.5,
                    label="Guidance Scale",
                    info="Contrôle combien l'image suit le prompt"
                )
            
            generate_btn = gr.Button(
                "🚀 Générer l'Image", 
                variant="primary",
                size="lg"
            )
            
            # Section d'exemples
            gr.Markdown("### 💡 Exemples rapides")
            for i, example in enumerate(example_prompts):
                gr.Button(
                    example, 
                    size="sm",
                    elem_classes="example-prompt"
                ).click(
                    lambda x=example: x,
                    outputs=prompt_input
                )
        
        with gr.Column(scale=1):
            # Section des résultats
            gr.Markdown("### 🖼️ Résultat")
            
            output_image = gr.Image(
                label="Image Générée",
                height=400,
                show_download_button=True
            )
            
            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                max_lines=3
            )
            
            # Bouton de téléchargement supplémentaire
            download_btn = gr.DownloadButton(
                "📥 Télécharger l'image",
                visible=False,
                size="sm"
            )
    
    # Section d'information
    with gr.Accordion("ℹ️ Informations et conseils", open=False):
        gr.Markdown("""
        ### 💡 Conseils pour de meilleurs résultats:
        
        - **Soyez descriptif**: "Un chat astronaut avec un casque doré dans l'espace étoilé"
        - **Ajoutez le style**: "style aquarelle", "dessin animé", "photo réaliste", "peinture à l'huile"
        - **Décrivez l'ambiance**: "lumière douce du coucher de soleil", "nuit étoilée", "brume matinale"
        - **Mentionnez les détails**: "textures détaillées", "couleurs vives", "arrière-plan flou"
        
        ### 🛠️ Fonctionnalités:
        - Génération d'images 512x512 pixels
        - Ajustement de la qualité via les paramètres
        - Téléchargement direct des images
        - Interface optimisée pour mobile et desktop
        
        ### ⚠️ Limitations:
        - Génération sur CPU (plus lent que GPU)
        - Qualité dépend de la description
        - Temps de génération: 30-60 secondes
        """)
    
    # Gestion des interactions
    def process_generation(prompt, num_steps, guidance_scale):
        """Traite la génération et prépare le téléchargement"""
        image, message = generate_image(prompt, num_steps, guidance_scale)
        
        if image is not None:
            # Sauvegarde pour le téléchargement
            file_path = save_image(image, prompt)
            return image, message, gr.DownloadButton(visible=True, value=file_path)
        else:
            return None, message, gr.DownloadButton(visible=False)
    
    # Connexion des événements
    generate_btn.click(
        fn=process_generation,
        inputs=[prompt_input, num_steps, guidance_scale],
        outputs=[output_image, status_output, download_btn]
    )
    
    # Entrée avec la touche Enter
    prompt_input.submit(
        fn=process_generation,
        inputs=[prompt_input, num_steps, guidance_scale],
        outputs=[output_image, status_output, download_btn]
    )

# Configuration du lancement
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Accessible depuis d'autres appareils
        server_port=7860,
        share=True,  # Crée un lien public
        show_error=True,
        debug=False
    )