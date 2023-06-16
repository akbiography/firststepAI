from transformers import GPT4LMHeadModel, GPT4Tokenizer

class ChatBot:
    def __init__(self, model_name='gpt4'):
        self.model = GPT4LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT4Tokenizer.from_pretrained(model_name)
        self.temperature = 0.7
        self.max_length = 100
        self.num_return_sequences = 1

    def generate_response(self, input_text):
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')

        output = self.model.generate(
            input_ids,
            max_length=self.max_length,
            num_return_sequences=self.num_return_sequences,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )

        responses = []
        for i in range(self.num_return_sequences):
            response = self.tokenizer.decode(output[i], skip_special_tokens=True)
            responses.append(response)

        return responses

    def start_conversation(self):
        print("Привіт! Я чат-бот. Я можу відповідати на ваші запитання. Давайте почнемо!")

        while True:
            try:
                user_input = input("Ваше запитання: ")
            except KeyboardInterrupt:
                print("\nДо побачення!")
                break

            if user_input.lower() == "пока":
                print("До побачення!")
                break

            self.process_user_input(user_input)

    def process_user_input(self, user_input):
        if user_input.lower() == "сохранить":
            self.save_model()
            print("Модель чат-бота успешно сохранена.")
        elif user_input.lower() == "загрузить":
            self.load_model()
            print("Модель чат-бота успешно загружена.")
        elif user_input.lower() == "параметры":
            self.print_model_parameters()
        elif user_input.lower() == "справка":
            self.print_help()
        elif user_input.lower() == "очистить":
            self.clear_history()
        elif user_input.lower() == "повтори":
            self.repeat_last_response()
        elif user_input.lower() == "помощь":
            self.get_assistance()
        elif user_input.lower() == "генерация":
            self.toggle_generation()
        elif user_input.lower() == "настройки":
            self.print_settings()
        else:
            self.generate_and_print_responses(user_input)

    def generate_and_print_responses(self, user_input):
        try:
            responses = self.generate_response(user_input)

            print("Відповіді:")
            for i, response in enumerate(responses):
                print(f"{i+1}. {response}")

        except Exception as e:
            print("Виникла помилка під час генерації відповіді:")
            print(e)

    def save_model(self, save_directory="chatbot_model"):
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    def load_model(self, load_directory="chatbot_model"):
        self.model = GPT4LMHeadModel.from_pretrained(load_directory)
        self.tokenizer = GPT4Tokenizer.from_pretrained(load_directory)

    def print_model_parameters(self):
        print("Параметри моделі:")
        for name, parameter in self.model.named_parameters():
            print(f"{name}: {parameter.shape}")

    def print_help(self):
        print("Доступні команди:")
        print("- 'сохранить': зберегти модель чат-бота")
        print("- 'загрузить': завантажити модель чат-бота")
        print("- 'параметры': вивести параметри моделі чат-бота")
        print("- 'справка': вивести цю довідку з командами")
        print("- 'очистить': очистити історію відповідей")
        print("- 'повтори': повторити останню відповідь")
        print("- 'помощь': отримати допомогу від бота")
        print("- 'генерация': увімкнути або вимкнути генерацію")
        print("- 'настройки': вивести поточні настройки бота")

    def clear_history(self):
        print("Історія відповідей очищена.")
        # Додайте код для очищення історії відповідей

    def repeat_last_response(self):
        print("Повтор останньої відповіді:")
        # Додайте код для повторення останньої відповіді

    def get_assistance(self):
        print("Бот надає допомогу. Що вас цікавить?")
        # Додайте код для отримання допомоги від бота

    def toggle_generation(self):
        print("Генерація відповідей увімкнена/вимкнена.")
        # Додайте код для увімкнення або вимкнення генерації відповідей

    def print_settings(self):
        print("Поточні настройки бота:")
        # Додайте код для виведення поточних настройок бота


bot = ChatBot()

bot.start_conversation()
