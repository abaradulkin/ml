# Импорт библиотек
from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import torch
import numpy as np


# Определяем устройство
if torch.cuda.is_available():
    device = "cuda"
else:
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Используется устройство: {device}")


# Загрузка датасета с шутками
dataset = load_dataset("ysharma/short_jokes")
# split_dataset = dataset["train"].train_test_split(test_size=0.1)  # 90% train, 10% validation

print("Длинна датасета", len(dataset["train"]))
print("Пример шутки:", dataset["train"][68])

# Загрузка токенизатора и модели
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# Предобработка данных - токенизация, подготовка инпутов/аутпутов

# Подготовка данных
def preprocess(examples):
    model_inputs = tokenizer(
        examples["Joke"],
        max_length=64,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    char_lengths = [len(joke) for joke in examples["Joke"]]
    # ВНИМАНИЕ: считаем реальные токены по attention_mask
    token_lengths = model_inputs["attention_mask"].sum(dim=1).tolist()

    return {'input_ids': model_inputs['input_ids'], 'attention_mask': model_inputs['attention_mask'], "char_length": char_lengths, "token_length": token_lengths}

tokenized_dataset = dataset.map(preprocess, batched=True, batch_size=256)
print(tokenized_dataset["train"][68])

def filter_short_examples(examples):
    result = [x >= 20 for x in examples["token_length"]]
    return result

# Применяем фильтр к нужному разбиению (обычно это "train")
filtered_dataset = tokenized_dataset.filter(filter_short_examples, batched=True)

print(f"Размер обучающей выборки: {len(tokenized_dataset['train'])} -> {len(filtered_dataset['train'])}")


# Предобработка данных - токенизация, подготовка инпутов/аутпутов
# Подготовка данных
def preprocess(examples):
    labels = examples["input_ids"]
    input_ids = [[3806, 10802, 10] + x[:10] + [0, 0, 0] for x in examples["input_ids"]]

    return {'input_ids': input_ids, 'labels': labels}


prepared_dataset = filtered_dataset.map(preprocess, batched=True, batch_size=256)
final_dataset = prepared_dataset["train"].train_test_split(test_size=0.1)  # 90% train, 10% validation
print(final_dataset['test'][1])


# Параметры обучения
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1, # показываем каждый пример по одному разу - можно увеличить, тогда будем обучаться дольше
    per_device_train_batch_size=32, # скармливаем модели 32 примеров параллельно
    gradient_accumulation_steps=4,  # Gradient Accumulation Steps позволяют имитировать большой размер батча без большого потребления памяти.
    per_device_eval_batch_size=1,
    learning_rate=3e-5,
    logging_steps=300, # каждые 300 итераций выводим лосс
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=False if device == torch.device("mps") else True, # формат хранения весов, для экономии памяти
    report_to="none",
    # optim="adamw_torch",
    # gradient_checkpointing=True
)


# Мониторим перплексию по ходу обучения
def compute_metrics(eval_pred):
    loss = trainer.evaluate(predict_loss_only=True)["eval_loss"]
    perplexity = np.exp(loss)
    return {"perplexity": perplexity}


# Инициализация Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=final_dataset["train"],
    eval_dataset=final_dataset["test"],
    compute_metrics=compute_metrics
)


# Обучение модели
trainer.train()
model.save_pretrained("./t5-jokes")
tokenizer.save_pretrained("./t5-jokes")
