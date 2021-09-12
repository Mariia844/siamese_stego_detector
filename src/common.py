import configparser

def get_config():
    config = configparser.ConfigParser()
    #config.read('../../config.ini')
    config.read('../config.ini')
    return config

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

from datetime import datetime
import telebot

config = get_config()['telegram']
bot = telebot.TeleBot(config['token'])

# class MessageProducer:
#     def __init__(self, scope: str, message_template : str  = '[{scope}] {time}: {message}', time_format : str = '%H:%M:%S.%f'):
#         self.scope = scope
#         self.message_template : str = message_template
#         self.time_format : str = time_format
#     def __get_formatted_message(self, message, **kwargs):
#         return self.message_template.format(
#             scope = self.scope,
#             time = datetime.now().strftime(self.time_format), 
#             message = message,
#             kwargs=kwargs)
#     def produce_message(self, message, **kwargs):
#         message = self.__get_formatted_message(message, kwargs=kwargs)
#         print(message)

# class TelegramMessageProducer(MessageProducer):
#     def __init__(self, scope: str, token: str, chat_id):
#         super().__init__(scope)
#         self.bot = telebot.TeleBot(token)
#         self.chat_id = chat_id
#     def produce_message(self, message, **kwargs):
#         self.bot.send_message(chat_id=self.chat_id, text=super().__get_formatted_message(message, kwargs=kwargs))
    