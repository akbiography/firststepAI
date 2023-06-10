from transformers import GPT4LMHeadModel, GPT4Tokenizer

# Завантаження попередньо навченої моделі та токенізатора
model_name = 'gpt4'  # або 'gpt4-medium', 'gpt4-large', 'gpt4-xl' для більш потужних моделей
model = GPT4LMHeadModel.from_pretrained(model_name)
tokenizer = GPT4Tokenizer.from_pretrained(model_name)

# Функція для генерації відповіді моделлю
def generate_response(input_text, max_length=100, num_return_sequences=1, temperature=0.7):
    # Токенізація тексту
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Генерація відповіді
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id  # Додатково додаємо токен завершення речення
    )

    # Декодування згенерованих відповідей
    responses = []
    for i in range(num_return_sequences):
        response = tokenizer.decode(output[i], skip_special_tokens=True)
        responses.append(response)

    return responses

# Початок діалогу
print("Привіт! Я чат-бот. Я можу відповідати на ваші запитання. Давайте почнемо!")

while True:
    # Введення користувача
    try:
        user_input = input("Ваше запитання: ")
    except KeyboardInterrupt:
        print("\nДо побачення!")
        break

    # Перевірка умови виходу з циклу
    if user_input.lower() == "пока":
        print("До побачення!")
        break

    # Параметри генерації відповіді
    max_length = 200  # Збільшуємо максимальну довжину згенерованої відповіді
    num_return_sequences = 2  # Збільшуємо кількість згенерованих відповідей
    temperature = 0.8  # Збільшуємо температуру генерації

    try:
        # Генерація відповіді
        responses = generate_response(user_input, max_length, num_return_sequences, temperature)

        # Виведення відповідей моделі
        print("Відповіді:")
        for i, response in enumerate(responses):
            print(f"{i+1}. {response}")

    except Exception as e:
        print("Виникла помилка під час генерації відповіді:")
        print(e)
        continue