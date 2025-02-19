from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Ruta al archivo output2.json
dataset_path = "output2.json"

# Cargar el dataset
dataset = load_dataset("json", data_files=dataset_path)

# Tokenizador y modelo base GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # O usa tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Si agregaste un token especial, redimensiona el modelo
model.resize_token_embeddings(len(tokenizer))

# Concatenar pregunta y respuesta en una sola entrada
def preprocess_function(examples):
	concatenated = [
        	f"Pregunta: {q} Respuesta: {a}"
        	for q, a in zip(examples["question"], examples["answer"])
	]
	tokenized = tokenizer(
        	concatenated,
        	truncation=True,
        	padding="max_length",
        	max_length=512
	)
    	# Hacer que los labels sean una copia de input_ids
	tokenized["labels"] = tokenized["input_ids"].copy()
	return tokenized

# Aplicar la tokenización
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["question", "answer"])

# Configurar el dataset para entrenamiento
train_dataset = tokenized_datasets["train"]

# Configuración del entrenamiento
training_args = TrainingArguments(
    output_dir="./fine_tuned_gpt2",
    evaluation_strategy="no",
    eval_steps=500,
    logging_dir="./logs",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    save_steps=1000,
    save_total_limit=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=50,
    learning_rate=5e-5,
    fp16=False,  # true -> Si tienes GPU con soporte FP16
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Afinar el modelo
trainer.train()

# Guardar el modelo afinado
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")
