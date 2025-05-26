# Импорт библиотек
from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import torch
import numpy as np


# Определяем устройство
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    # device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    device = torch.device("cpu")
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
    # Токенизируем каждую шутку отдельно
    input_ids = []
    attention_masks = []

    for joke in examples["Joke"]:
        # Токенизируем каждую шутку без padding
        encoded = tokenizer(
            joke,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
        input_ids.append(encoded["input_ids"][0].tolist())
        attention_masks.append(encoded["attention_mask"][0].tolist())

    # Считаем длины токенов
    token_lengths = [len(ids) for ids in input_ids]
    char_lengths = [len(joke) for joke in examples["Joke"]]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'char_length': char_lengths,
        'token_length': token_lengths
    }


tokenized_dataset = dataset.map(preprocess, batched=True, batch_size=256)
print(tokenized_dataset["train"][68])
token_lengths = tokenized_dataset["train"]["token_length"]
print(f"Минимальное количество токенов: {min(token_lengths)}")
print(f"Максимальное количество токенов: {max(token_lengths)}")
print(f"Среднее количество токенов: {sum(token_lengths) / len(token_lengths)}")


def filter_short_examples(examples):
    result = [64 >= x >= 13 for x in examples["token_length"]]
    return result


# Применяем фильтр к нужному разбиению (обычно это "train")
filtered_dataset = tokenized_dataset.filter(filter_short_examples, batched=True)

print(f"Размер обучающей выборки: {len(tokenized_dataset['train'])} -> {len(filtered_dataset['train'])}")


# Предобработка данных - токенизация, подготовка инпутов/аутпутов
# Подготовка данных
def preprocess(examples):
    labels = examples["input_ids"]
    inputs = ["generate joke: " + text for text in examples["Joke"]]

    # Токенизация входов (промптов)
    # на вход модели будем подавать завязку шутки
    model_inputs = tokenizer(
        inputs,
        max_length=16,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    # на выход ожидаем получить всю шутку целиком
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["Joke"],
            max_length=64,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


prepared_dataset = filtered_dataset.map(preprocess, batched=True, batch_size=256)
final_dataset = prepared_dataset["train"].train_test_split(test_size=0.1)  # 90% train, 10% validation
print(final_dataset['test'][1])


# Параметры обучения
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,  # показываем каждый пример по одному разу - можно увеличить, тогда будем обучаться дольше
    per_device_train_batch_size=32,  # скармливаем модели 32 примеров параллельно
    per_device_eval_batch_size=1,
    learning_rate=3e-5,
    logging_steps=100,  # каждые 300 итераций выводим лосс
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=False,  # формат хранения весов, для экономии памяти
    report_to="none"
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

final_loss = trainer.evaluate()["eval_loss"]
if final_loss < 2.0:
    model.save_pretrained("./t5-jokes")
    tokenizer.save_pretrained("./t5-jokes")
