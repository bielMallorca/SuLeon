import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ruta del modelo
MODEL_PATH = "fine_tuned_gpt2"  # Ajusta esta ruta según sea necesario

# Cargar el tokenizador y el modelo
print("Cargando el modelo...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.eval()
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None
    tokenizer = None

def generar_respuesta(texto):
    """
    Genera una respuesta basada en el texto de entrada usando el modelo entrenado.

    :param texto: Texto de entrada (prompt).
    :return: Respuesta generada por el modelo.
    """
    if model is None or tokenizer is None:
        return "Error: el modelo no está cargado correctamente."

    inputs = tokenizer(texto, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=200, num_return_sequences=1, do_sample=True, temperature=0.7)

    respuesta = tokenizer.decode(output[0], skip_special_tokens=True)
    return respuesta
