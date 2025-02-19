import requests
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import whisper
import ffmpeg

app = Flask(__name__)

# Configuración de la API de SuLeon
#API_URL = "https://api.mygpts.com/suleon"  # Reemplaza con la URL correcta

# Cargar el modelo Whisper
modelwi = whisper.load_model("base")

# Función para convertir WebM a WAV
def convert_webm_to_wav(input_path, output_path):
    try:
        ffmpeg.input(input_path).output(output_path, format="wav", ar=16000, ac=1).run(overwrite_output=True)
        return True
    except ffmpeg.Error as e:
        print(f"Error during conversion: {e.stderr.decode()}")
        return False

@app.route("/")
def home():
    return render_template("index.html")
# Endpoint principal para transcribir y enviar a SuLeon
@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    input_path = "temp_audio.webm"
    output_path = "temp_audio.wav"
    audio_file.save(input_path)
    print("pasando audio a web")
    # Convertir el archivo WebM a WAV
    if not convert_webm_to_wav(input_path, output_path):
        return jsonify({"error": "Audio conversion failed"}), 500
    print("iniciando transcripcion")
    try:
        # Transcribir el audio usando Whisper
        result = modelwi.transcribe(output_path, language="en")
        transcription = result["text"]
#        transcription = "hello, how are you?"
        print("----- "+transcription)
        # Enviar la transcripción a la API de SuLeon
        # Cargar el modelo y tokenizador ajustados
        model_path = "./mi_llm_lib/fine_tuned_gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
	# Asegurar que el pad_token esté configurado
        tokenizer.pad_token = tokenizer.eos_token
        input_text = transcription
        print (transcription)
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        print("generar texto")
	# Generar texto pasando la atención explícitamente
        outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs["attention_mask"],  # Incluye la atención
                max_length=50,
                pad_token_id=tokenizer.pad_token_id,  # Asegurar que el pad_token_id esté configurado
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True
        )
	# Decodificar el texto generado
        print("texto generado")
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Generated Text:", generated_text)
        return jsonify({
                "transcription": transcription,
                "suleon_response": generated_text
            })

    except Exception as e:
        return jsonify({"error": f"Error during transcription: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, ssl_context=("cert.pem", "key.pem"))
